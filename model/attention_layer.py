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
"""Implementation of multiheaded attention and self-attention layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Attention(tf.layers.Layer):
  """Multi-headed attention layer."""

  def __init__(self, hidden_size, num_heads, attention_dropout, train):
    if hidden_size % num_heads != 0: # hidden_size必须整除num_heads
      raise ValueError("Hidden size must be evenly divisible by the number of "
                       "heads.")

    super(Attention, self).__init__()
    self.hidden_size = hidden_size # 这3个是参数
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout
    self.train = train # 这个是啥，函数么

    # Layers for linearly projecting the queries, keys, and values.
    self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q") # 定义一个线性映射层
    self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
    self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")

    self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                              name="output_transform")

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.

    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.

    Args:
      x: A tensor with shape [batch_size, length, hidden_size]

    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
    with tf.name_scope("split_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      # Calculate depth of last dimension after it has been split.
      depth = (self.hidden_size // self.num_heads) # 每个head分得多少hidden_size

      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, self.num_heads, depth]) # 将最后一个维度分成了两个维度即：hidden_size --> num_heads * depth

      # Transpose the result
      return tf.transpose(x, [0, 2, 1, 3]) # 将num_heads那一维放到batch_size后面

  def combine_heads(self, x):
    """Combine tensor that has been split.

    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    with tf.name_scope("combine_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[2]
      x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
      return tf.reshape(x, [batch_size, length, self.hidden_size])

  def call(self, x, y, bias, cache=None): # 在x和y之间建立attention机制
    """Apply attention mechanism to x and y.

    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      bias: attention bias that will be added to the result of the dot product. # 在点积结果上添加的偏差项
      cache: (Used during prediction) dictionary with tensors containing results # 还不知道是干啥
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.

    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
    # Linearly project the query (q), key (k) and value (v) using different
    # learned projections. This is in preparation of splitting them into
    # multiple heads. Multi-head attention uses multiple queries, keys, and
    # values rather than regular attention (which uses a single q, k, v).
    q = self.q_dense_layer(x) # x作为query，先来个线性映射
    k = self.k_dense_layer(y) # y作为key，也来个线性映射
    v = self.v_dense_layer(y) # y作为value，再来个线性映射

    if cache is not None: # 如果cache有，则从里面加载k和v，与现有kv拼接，但不知道这是在干啥
      # Combine cached keys and values with new keys and values.
      k = tf.concat([cache["k"], k], axis=1) # 拼完之后变长了？
      v = tf.concat([cache["v"], v], axis=1)

      # Update cache
      cache["k"] = k
      cache["v"] = v

    # Split q, k, v into heads.
    q = self.split_heads(q) # 将q分多个头，从[batch_size, length, hidden_size] --> [batch_size, self.num_heads, length, depth]
    k = self.split_heads(k) # 做的事情与上面一样
    v = self.split_heads(v)

    # Scale q to prevent the dot product between q and k from growing too large.
    depth = (self.hidden_size // self.num_heads)
    q *= depth ** -0.5

    # Calculate dot product attention
    logits = tf.matmul(q, k, transpose_b=True)
    logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    if self.train:
      weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
    attention_output = tf.matmul(weights, v)

    # Recombine heads --> [batch_size, length, hidden_size]
    attention_output = self.combine_heads(attention_output)

    # Run the combined outputs through another linear projection layer.
    attention_output = self.output_dense_layer(attention_output)
    return attention_output


class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def call(self, x, bias, cache=None):
    return super(SelfAttention, self).call(x, x, bias, cache)
