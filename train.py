import itertools

from flax import linen as nn
from flax.training import train_state
import jax.numpy as jnp
import jax
import numpy as np
import optax
from absl import app
from absl import flags
import datasets
from tqdm import tqdm
import pandas as pd

import models

DATA_FILE="data/ETTm1.csv.gz"

FLAGS = flags.FLAGS

flags.DEFINE_integer('hidden_size', 256, 'Size of the hidden state')
flags.DEFINE_integer('memory_size', 128, 'Size of the memory')

flags.DEFINE_integer('batch_size', 64, 'Batch size')

flags.DEFINE_integer('num_epochs', 500, 'Number of steps')

flags.DEFINE_integer("eval_every", 1 , "Evaluate every N steps")

flags.DEFINE_integer('sequence_size', 128, 'Size of the input sequence')
flags.DEFINE_float('dt', 1., 'Time step')  

flags.DEFINE_integer('predict_offset', 16, 'Offset to predict')

flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')

flags.DEFINE_string('model_dir', 'model', 'Directory to save model')

flags.DEFINE_boolean('use_lstm', False, 'Use LSTM instead of LMU')


def get_model(hidden_size, memory_size, input_size, use_lstm=False):
  if use_lstm:
    return models.LSTM(
      cell=nn.LSTMCell(hidden_size),
      output_size=input_size)
  else:
    cell=models.LMUCell(
      hidden_size=hidden_size,
      memory_size=memory_size,
      input_size=input_size,
      theta=FLAGS.sequence_size,
      dt=FLAGS.dt,
      )

    return models.LMU(cell=cell, output_size=input_size)


def get_model_and_params(
    key, batch_size, sequence_size, hidden_size, memory_size, input_size, use_lstm=False):
  model = get_model(hidden_size, memory_size, input_size, use_lstm=use_lstm)
  params = model.init(key, jnp.ones((batch_size, sequence_size, input_size)))
  return model, params


def get_train_state(key, input_size):
  batch_size = FLAGS.batch_size
  sequence_size = FLAGS.sequence_size
  hidden_size = FLAGS.hidden_size
  memory_size = FLAGS.memory_size
  use_lstm = FLAGS.use_lstm

  model, params = get_model_and_params(
    key, batch_size, sequence_size, hidden_size, memory_size, input_size,
    use_lstm=use_lstm)

  tx = optax.adam(FLAGS.learning_rate)

  return train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx)


def mse_loss_fn(predicted, targets):
  return jnp.mean((predicted - targets)**2)


def compute_metrics(predicted, targets):
  return jnp.mean((predicted - targets)**2) 


@jax.jit
def train_step(state, batch):
  def loss_fn(params):
    predicted = state.apply_fn(params, batch['inputs'])
    return mse_loss_fn(predicted, batch['targets'])

  grad_fn = jax.value_and_grad(loss_fn)

  loss, grad = grad_fn(state.params)

  # jax.debug.print('loss={loss}, grad={grad}', loss=loss, grad=grad)

  # jax.debug.print('state before={state}', state=state)

  state = state.apply_gradients(grads=grad)

  # jax.debug.print('state after={state}', state=state)

  return state, loss


@jax.jit
def eval_step(state, batch):
  predicted = state.apply_fn(state.params, batch['inputs'])
  return compute_metrics(predicted, batch['targets'])


def train_and_eval(train_df, test_df):
  rng = jax.random.PRNGKey(0)
  state = get_train_state(rng, train_df.input_size)

  eval_every = FLAGS.eval_every

  train_loss = test_loss = 0

  for epoch in range(FLAGS.num_epochs):
    train_loss = 0

    for batch in tqdm(train_df):

      state, loss = train_step(state, batch)

      train_loss += loss

    if eval_every > 0 and epoch % eval_every == 0:
      test_loss = 0

      for batch in tqdm(test_df):
        metrics = eval_step(state, batch)
        test_loss += metrics

      train_loss /= len(train_df)
      test_loss /= len(test_df)

      print(f'Epoch {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}')


class DataGenerator:
  def __init__(self, data, sequence_size, predict_offset, y_column, x_columns=None,
               randomized=False, batch_size=1):
    self.data = data
    self.sequence_size = sequence_size
    self.predict_offset = predict_offset
    self.randomized = randomized
    self.batch_size = batch_size
    self.y_column = y_column

    if x_columns is None:
      x_columns = [col for col in data.columns]

    self.x_columns = x_columns

    assert y_column in x_columns

    self.y_column_idx = x_columns.index(y_column)

    self.input_size = len(x_columns)

    print("Got x_columns", x_columns, "y_column", y_column, "y_column_idx", self.y_column_idx, "input_size", self.input_size  )

  def __iter__(self):
    iter = create_sequential_iterator(
      self.data[self.x_columns].values, self.sequence_size,
      self.predict_offset, y_column=self.y_column_idx,
      randomized=self.randomized)

    return batch_iter(iter, self.batch_size)

  def __len__(self):
    n = len(self.data) - self.sequence_size - self.predict_offset

    return n // self.batch_size


def create_sequential_iterator(
    data, sequence_size, predict_offset, y_column,
    randomized=False):
  n = len(data)
  delta = sequence_size + predict_offset

  if randomized:
    indices = np.random.permutation(n - delta)
  else:
    indices = range(0, n - delta)

  for i in indices:
    input = data[i:i+sequence_size, :]
    target = data[i + delta: i + delta + 1, y_column:y_column+1]

    yield {'inputs': input, 'targets': target}


def batch_iter(iterator, batch_size):
  batch = []
  for example in iterator:
    batch.append(example)
    if len(batch) == batch_size:
      inputs = jnp.array([ex['inputs'] for ex in batch])
      targets = jnp.array([ex['targets'] for ex in batch])
      yield {'inputs': inputs, 'targets': targets}
      batch = []

  if batch:
    inputs = jnp.array([ex['inputs'] for ex in batch])
    targets = jnp.array([ex['targets'] for ex in batch])
    yield {'inputs': inputs, 'targets': targets}


def make_sin_cos(x, period):
  x = 2 * np.pi * x / period
  return np.sin(x), np.cos(x)


def parse_data():
  df = pd.read_csv(DATA_FILE)

  df.date = pd.to_datetime(df.date)

  df['month_sin'], df['month_cos'] = make_sin_cos(df.date.dt.month, 12)
  df['day_sin'], df['day_cos'] = make_sin_cos(df.date.dt.day, 31)
  df['hour_sin'], df['hour_cos'] = make_sin_cos(df.date.dt.hour, 24)

  df = df.drop(columns=['date'])

  return df


def main(argv):
  data = parse_data()

  float_columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

  mean, std = data[float_columns].mean(), data[float_columns].std()

  data[float_columns] = (data[float_columns] - mean) / std

  n_train = int(len(data) * 0.6)

  train_data = data[:n_train]
  test_data = data[n_train:]

  train_iter = DataGenerator(
    train_data, FLAGS.sequence_size, FLAGS.predict_offset,
    y_column='OT', randomized=True, batch_size=FLAGS.batch_size)

  test_iter = DataGenerator(
    test_data, FLAGS.sequence_size, FLAGS.predict_offset,
    y_column='OT', randomized=False, batch_size=FLAGS.batch_size)

  print("Got data lens", len(train_iter), len(test_iter))

  train_and_eval(train_iter, test_iter)


if __name__ == '__main__':
  app.run(main)