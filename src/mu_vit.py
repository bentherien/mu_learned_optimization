# Copyright 2023 Google LLC.
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

from typing import Any, Callable, Optional, Tuple, Type

import flax.linen as nn
import jax.numpy as jnp

from vit_jax import models_resnet


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x):
    return x


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    posemb_init: positional embedding initializer.
  """

  posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]

  @nn.compact
  def __call__(self, inputs):
    """Applies the AddPositionEmbs module.

    Args:
      inputs: Inputs to the layer.

    Returns:
      Output tensor with shape `(bs, timesteps, in_dim)`.
    """
    # inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape)
    return inputs + pe


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  hidden_size: int
  mlp_dim: int
  dtype: Dtype = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        features=self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        features=actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            x)
    output = nn.Dropout(
        rate=self.dropout_rate)(
            output, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    inputs: input data.
    mlp_dim: dimension of the mlp on top of attention block.
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout for attention heads.
    deterministic: bool, deterministic or not (to apply dropout).
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
  """

  hidden_size: int
  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.

    Returns:
      output after transformer encoder block.
    """

    # Attention block.
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    # Default init is fine for MuP
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=jax.nn.initializers.truncated_normal(1/jnp.sqrt(self.hidden_size)),
        bias_init=jax.nn.initializers.normal(1),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads)(
            x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    # Default init is fine for MuP
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(
        hidden_size=self.hidden_size,
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate,
        #MuP init
        kernel_init=jax.nn.initializers.truncated_normal(1/jnp.sqrt(self.hidden_size)),
        bias_init=jax.nn.initializers.normal(1),)(
            y, deterministic=deterministic)

    return x + y


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    num_layers: number of layers
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate in self attention.
  """

  hidden_size: int
  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  add_position_embedding: bool = True

  @nn.compact
  def __call__(self, x, *, train):
    """Applies Transformer model on the inputs.

    Args:
      x: Inputs to the layer.
      train: Set to `True` when training.

    Returns:
      output of a transformer encoder.
    """
    assert x.ndim == 3  # (batch, len, emb)

    if self.add_position_embedding:
      x = AddPositionEmbs(
          # posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.

          # in MuP PosEmbs map finite --> infinite
          # thus they are treated as input weights
          posemb_init=jax.nn.initializers.truncated_normal(1/jnp.sqrt(self.hidden_size)), 
          name='posembed_input')(
              x)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          hidden_size=self.hidden_size,
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          num_heads=self.num_heads)(
              x, deterministic=not train)
      
    # Default init is fine for MuP
    encoded = nn.LayerNorm(name='encoder_norm')(x)

    return encoded

#mup lrs
# {'mu_mlp': 
#  {'mup_lrs': 
#   {'~/linear_0': 
#    {'w': 1.0, 'b': 1.0}, 
#    '~/linear_1': {'w': 0.0078125, 'b': 1.0}, 
#    '~/linear_2': {'w': 0.0078125, 'b': 1.0}, 
#    '~/linear_3': {'w': 1.0, 'b': 1.0}}, 
#    'layer_0_act_l1': Array(1.1220654, dtype=float32), 
#    'layer_1_act_l1': Array(1.0116022, dtype=float32), 
#    'layer_2_act_l1': Array(1.0860213, dtype=float32), 
#    'layer_3_act_l1': Array(0.7819562, dtype=float32)}}

class MuVisionTransformer(nn.Module):
  """VisionTransformer."""

  num_classes: int
  patches: Any
  transformer: Any
  hidden_size: int
  resnet: Optional[Any] = None
  representation_size: Optional[int] = None
  classifier: str = 'token'
  head_bias_init: float = 0.
  encoder: Type[nn.Module] = Encoder
  model_name: Optional[str] = None

  @nn.compact
  def __call__(self, inputs, *, train):

    x = inputs
    # (Possibly partial) ResNet root.
    if self.resnet is not None:
      width = int(64 * self.resnet.width_factor)

      # Root block.
      x = models_resnet.StdConv(
          features=width,
          kernel_size=(7, 7),
          strides=(2, 2),
          use_bias=False,
          name='conv_root')(
              x)
      x = nn.GroupNorm(name='gn_root')(x)
      x = nn.relu(x)
      x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

      # ResNet stages.
      if self.resnet.num_layers:
        x = models_resnet.ResNetStage(
            block_size=self.resnet.num_layers[0],
            nout=width,
            first_stride=(1, 1),
            name='block1')(
                x)
        for i, block_size in enumerate(self.resnet.num_layers[1:], 1):
          x = models_resnet.ResNetStage(
              block_size=block_size,
              nout=width * 2**i,
              first_stride=(2, 2),
              name=f'block{i + 1}')(
                  x)

    n, h, w, c = x.shape

    print(x.shape,'before conv')
    # We can merge s2d+emb into a single conv; it's the same.
    x = nn.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding',
        # MuP input layer init
        kernel_init=jax.nn.initializers.truncated_normal(1/jnp.sqrt(self.hidden_size)),
        bias_init=jax.nn.initializers.normal(1))(
            x )

    print(x.shape,'after conv')
    # exit(0)

    # Here, x is a grid of embeddings.

    # (Possibly partial) Transformer.
    if self.transformer is not None:
      n, h, w, c = x.shape
      x = jnp.reshape(x, [n, h * w, c])

      # If we want to add a class token, add it here.
      if self.classifier in ['token', 'token_unpooled']:
        cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
        cls = jnp.tile(cls, [n, 1, 1])
        x = jnp.concatenate([cls, x], axis=1)

      x = self.encoder(hidden_size=self.hidden_size,name='Transformer', **self.transformer)(x, train=train)

    if self.classifier == 'token':
      x = x[:, 0]
    elif self.classifier == 'gap':
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
    elif self.classifier in ['unpooled', 'token_unpooled']:
      pass
    else:
      raise ValueError(f'Invalid classifier={self.classifier}')

    if self.representation_size is not None:
      x = nn.Dense(features=self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = IdentityLayer(name='pre_logits')(x)

    if self.num_classes:
      x = nn.Dense(
          features=self.num_classes,
          name='head',
          # MuP ouput layer init
          kernel_init=nn.initializers.zeros,
          bias_init=jax.nn.initializers.normal(1)
          )(x)
      
    return x * (1 / self.hidden_size) # MuP ouput multiplier







"""Vision Transformers based on the `vision_transformer` package!

See: https://github.com/google-research/vision_transformer for more info.
"""
from typing import Any, Tuple

import chex
import gin
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
import numpy as onp

import functools

Params = Any
ModelState = Any
PRNGKey = jnp.ndarray

def find_smallest_divisor(x,b):
  # Start from the smallest possible divisor greater than 1
  for y in range(2, x + 1):  # We start from 2 as 1 will always divide x and result in a itself
      if x % y == 0:  # Check if y is a divisor of x
          a = x // y  # Calculate a as the quotient of x divided by y
          if a < b:  # Check if a meets the condition
              return y  # Return the smallest y that meets the condition
  print("Warning: No smaller divisor found. Returning the original value.")
  return x  # Return x if no smaller divisor is found satisfying the condition

@functools.partial(jax.jit)
def multi_batch_forward(module, params, data, key):
  return jnp.concatenate( # pylint: disable=g-complex-comprehension
          [module.apply(params, chunk, train=False, rngs={"dropout": key}) for chunk in data["image"]])

class MuVisionTransformerTask(base.Task):
  """Vision Transformer task."""

  def __init__(self, datasets, cfg):
    num_c = datasets.extra_info["num_classes"]
    self.flax_module = MuVisionTransformer(num_classes=num_c, **cfg)
    self.datasets = datasets
    self.mup_lrs = None

  def init(self, key: chex.PRNGKey):
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    return self.flax_module.init({
        "params": key,
        "dropout": key
    },
      batch["image"],
      train=True)

  def mup_lrs_from_params(self, params):
    d = jax.tree_map(lambda x: x.shape, params)
    mup_lrs = jax.tree_map(lambda x: 1.0 ,params)
    for k in d['params']['Transformer'].keys():
      if 'encoderblock' in k:
          for kk,v in d['params']['Transformer'][k].items():
              if 'MlpBlock' in kk:
                  mup_lrs['params']['Transformer'][k][kk]['Dense_0']['kernel'] = 1/v['Dense_0']['kernel'][0]
                  mup_lrs['params']['Transformer'][k][kk]['Dense_1']['kernel'] =  1/v['Dense_1']['kernel'][0]
                  # print(kk, d['params']['Transformer'][k][kk].keys())
                  # for key in d['params']['Transformer'][k][kk].keys():
                  #     print(key, d['params']['Transformer'][k][kk][key].keys())
                  #     for kkey,vv in d['params']['Transformer'][k][kk][key].items():
                  #         print(key, kkey, vv)
                  #         print(mup_lrs['params']['Transformer'][k][kk][key][kkey])
                      
                      
              
              elif 'MultiHeadDotProductAttention' in kk:
                  mup_lrs['params']['Transformer'][k][kk]['key']['kernel'] = 1/v['key']['kernel'][0]
                  mup_lrs['params']['Transformer'][k][kk]['value']['kernel'] = 1/v['value']['kernel'][0]
                  mup_lrs['params']['Transformer'][k][kk]['query']['kernel'] = 1/v['query']['kernel'][0]
                  mup_lrs['params']['Transformer'][k][kk]['out']['kernel'] = 1/v['out']['kernel'][1] # input weight is in second dim
                  # print(kk, d['params']['Transformer'][k][kk].keys())
                  # for key in d['params']['Transformer'][k][kk].keys():
                  #     print(key, d['params']['Transformer'][k][kk][key].keys())
                  #     for kkey,v in d['params']['Transformer'][k][kk][key].items():
                  #         print(key, kkey, vv)
                  #         print('muplr =',mup_lrs['params']['Transformer'][k][kk][key][kkey])
    return mup_lrs
  
  def init_with_state(self, key: PRNGKey) -> Tuple[Params, ModelState]:
    params = self.init(key)
    if self.mup_lrs == None:
      self.mup_lrs = self.mup_lrs_from_params(params)
    state = self.mup_lrs

    # print(jax.tree_map(lambda x: x.shape, params))
    # print(jax.tree_map(lambda x: 1.0 ,params))
    # exit(0)
    return params, state

  def loss(self, params: Any, key: chex.PRNGKey, data: Any):
    logits = self.flax_module.apply(
        params, data["image"], train=True, rngs={"dropout": key})
    labels_onehot = jax.nn.one_hot(data["label"], logits.shape[1])
    loss_vec = base.softmax_cross_entropy(logits=logits, labels=labels_onehot)
    return jnp.mean(loss_vec)
  


  def loss_with_state(self, params: Any, state: Any, key: chex.PRNGKey, data: Any):
    logits = self.flax_module.apply(
        params, data["image"], train=True, rngs={"dropout": key})
    labels_onehot = jax.nn.one_hot(data["label"], logits.shape[1])
    loss_vec = base.softmax_cross_entropy(logits=logits, labels=labels_onehot)
    return jnp.mean(loss_vec), state
  
  def loss_and_accuracy(self, params: Params, key: PRNGKey, data: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:  # pytype: disable=signature-mismatch  # jax-ndarray
    num_classes = self.datasets.extra_info["num_classes"]

    threshold = 25000
    if data["image"].shape[0] > threshold:
      # If the batch is too large, we split it into smaller chunks.
      # This is to avoid running out of memory.
      # This is not necessary for the task to work, but it is useful for
      # large batch sizes.
      data["image"] = jnp.array_split(data["image"], 
                                      find_smallest_divisor(data["image"].shape[0],threshold), 
                                      axis=0)
      
      # logits = multi_batch_forward(self.flax_module,params, data, key)
      # print(jax.tree_map(lambda x: x.shape, data["image"]))
      logits = jnp.concatenate( # pylint: disable=g-complex-comprehension
          [self.flax_module.apply(params, chunk, train=False, rngs={"dropout": key}) for chunk in data["image"]])
    else:
      logits = self.flax_module.apply(params, data["image"], train=False, rngs={"dropout": key})
    
    # Calculate the loss as before
    labels = jax.nn.one_hot(data["label"], num_classes)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    loss = jnp.mean(vec_loss)
    
    # Calculate the accuracy
    predictions = jnp.argmax(logits, axis=-1)
    actual = data["label"]
    correct_predictions = predictions == actual
    accuracy = jnp.mean(correct_predictions.astype(jnp.float32))
    
    return loss, accuracy

  def normalizer(self, loss):
    max_class = onp.log(2 * self.datasets.extra_info["num_classes"])
    loss = jnp.nan_to_num(
        loss, nan=max_class, neginf=max_class, posinf=max_class)
    # shift to [0, 10] then clip.
    loss = 10 * (loss / max_class)
    return jnp.clip(loss, 0, 10)