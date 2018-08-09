# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of fully connected network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class FeedFowardNetwork(tf.layers.Layer):
  """Fully connected feedforward network."""

  def __init__(self, hidden_size, filter_size, relu_dropout, train, allow_pad):
    super(FeedFowardNetwork, self).__init__()
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout = relu_dropout
    self.train = train
    self.allow_pad = allow_pad

    self.filter_dense_layer = tf.layers.Dense( # 一个带激活函数的前向传播层
        filter_size, use_bias=True, activation=tf.nn.relu, name="filter_layer")
    self.output_dense_layer = tf.layers.Dense( # 一个线性映射层
        hidden_size, use_bias=True, name="output_layer")

  def call(self, x, padding=None):
    """Return outputs of the feedforward network.
    # 前向传播层有两个子层：
      (1) 带激活函数的前向传播层
      (2) 线性映射层

    # 为什么在前向传播之前需要把padding都去掉呢？是为了减小耗时？

    Args:
      x: tensor with shape [batch_size, length, hidden_size]
      padding: (optional) If set, the padding values are temporarily removed
        from x (provided self.allow_pad is set). The padding values are placed
        back in the output tensor in the same locations.
        shape [batch_size, length]
        记录着padding的信息，即padding部分都是1，非padding部分都是0

    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, hidden_size]
    """
    padding = None if not self.allow_pad else padding

    # Retrieve dynamically known shapes
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]

    if padding is not None:
      with tf.name_scope("remove_padding"): 
        # Flatten padding to [batch_size*length]
        pad_mask = tf.reshape(padding, [-1]) # 2维拉平成1维, 长度为batch_size*length

        nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9)) # 找到padding值是0即非padding的部分的位置，长度小于batch_size*length

        # Reshape x to [batch_size*length, hidden_size] to remove padding
        x = tf.reshape(x, [-1, self.hidden_size]) # 把 inputs的前两维也拉平，长度是batch_size*length
        x = tf.gather_nd(x, indices=nonpad_ids) # 从x中抽取片，组成nonpad_ids指定的形状，长度小于batch_size*length

        # Reshape x from 2 dimensions to 3 dimensions.
        x.set_shape([None, self.hidden_size])
        x = tf.expand_dims(x, axis=0) # 形状变为 [1, 小于batch_size*length, hidden_size]，为什么要添加一个维度呢？

    output = self.filter_dense_layer(x) # 先经过带激活层的前向网络
    if self.train:
      output = tf.nn.dropout(output, 1.0 - self.relu_dropout)
    output = self.output_dense_layer(output) # 再经过一个线性映射

    if padding is not None:
      with tf.name_scope("re_add_padding"):
        output = tf.squeeze(output, axis=0) # 去掉第一个维度
        output = tf.scatter_nd( # 把集中的非padding重新散步到各自原来的位置
            indices=nonpad_ids,
            updates=output,
            shape=[batch_size * length, self.hidden_size]
        )
        output = tf.reshape(output, [batch_size, length, self.hidden_size]) # 重新变回原来的形状
    return output
