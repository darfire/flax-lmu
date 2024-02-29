from typing import (
  Tuple,
)

from flax import linen as nn
from flax.linen.module import nowrap
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from flax.typing import Initializer
from flax.linen import initializers
from scipy.signal import cont2discrete

Array = jax.Array


def initialize_A(memory_size: int, theta: int) -> Array:
  A = np.zeros((memory_size, memory_size))
  for i in range(memory_size):
    for j in range(memory_size):
      if i < j:
        A[i, j] = -(2 * i + 1)
      else:
        A[i, j] = (2 * i + 1) * (-1)**(i - j + 1) 

  return A / theta


def initialize_B(memory_size: int, theta: int) -> Array:
  B = np.zeros((memory_size,))

  for i in range(memory_size):
    B[i] = (2 * i + 1) * (-1)**i

  return B.reshape((-1, 1)) / theta


class LMUCell(nn.RNNCellBase):
  input_size: int
  hidden_size: int
  memory_size: int

  theta: int

  dt: float = 1.

  carry_init: Initializer = initializers.xavier_normal()

  def setup(self):
    self.Wh = self.param(
      'Wh',
      lambda rng, shape: initializers.xavier_normal()(rng, shape),
      (self.hidden_size, self.hidden_size))

    self.Wx = self.param(
      'Wx',
      lambda rng, shape: initializers.xavier_normal()(rng, shape),
      (self.input_size, self.hidden_size))

    self.Wm = self.param(
      'Wm',
      lambda rng, shape: initializers.xavier_normal()(rng, shape),
      (self.memory_size, self.hidden_size))

    self.ex = self.param(
      'ex',
      initializers.lecun_uniform(),
      (self.input_size, 1))

    self.em = self.param(
      'em',
      lambda rng, shape: jnp.zeros(shape),
      (self.memory_size, 1))
    
    self.eh = self.param(
      'eh',
      initializers.lecun_uniform(),
      (self.hidden_size, 1))

    A = initialize_A(self.memory_size, self.theta)
    B = initialize_B(self.memory_size, self.theta)

    lti = (A, B, np.zeros((1, self.memory_size)), np.zeros((1, 1)))

    (A, B, _, _, _) = cont2discrete(lti, self.dt)

    self.A_ = jnp.asarray(A, jnp.float32)
    self.B_ = jnp.asarray(B, jnp.float32)

    # self.eps = self.param(
    #   'eps',
    #   lambda rng, shape: jax.random.uniform(rng, shape, minval=-1, maxval=1),
    #   (1, ))

  def __call__(self, carry, x):
    # x has the shape (batch, input_size)
    h, m = carry

    # jax.debug.print('carry m={m}, x={x}, h={h}', m=m, x=x, h=h)

    u = x @ self.ex + m @ self.em + h @ self.eh

    # jax.debug.print('computing m A_={A_}, u={u}, B_={B_}', A_=self.A_, u=u, B_=self.B_)

    # update the memory
    m = (self.A_ @ m.T).T + (self.B_ @ u.T).T

    # update the hidden state
    h = jnp.tanh(h @ self.Wh + x @ self.Wx + m @ self.Wm)

    # jax.debug.print('new h={h}, new m={m}', h=h, m=m)

    return (h, m), h

  @nowrap
  def initialize_carry(self, rng, input_shape: Tuple[int, ...]) -> Tuple[Array, Array]:
    key1, key2 = jax.random.split(rng)

    batch_dims = input_shape[:-1]
    key1, key2 = random.split(rng)

    h = self.carry_init(key1, batch_dims + (self.hidden_size,), jnp.float32)
    m = self.carry_init(key2, batch_dims + (self.memory_size,), jnp.float32)

    # jax.debug.print('initialize_carry h={h}, m={m}', h=h, m=m)
    
    return (h, m)

  @property
  def num_feature_axes(self):
    return 1


class LMU(nn.Module):
  cell: LMUCell
  output_size: int

  def setup(self):
    self.rnn = nn.RNN(self.cell)
    self.dense = nn.Dense(self.output_size)

  def __call__(self, inputs):
    outputs = self.rnn(inputs)

    return self.dense(outputs[:, -1:, :])


class LSTM(nn.Module):
  cell: nn.LSTMCell
  output_size: int

  def setup(self):
    self.rnn = nn.RNN(self.cell)
    self.dense = nn.Dense(self.output_size)

  def __call__(self, inputs):
    outputs = self.rnn(inputs)

    return self.dense(outputs[:, -1:, :])