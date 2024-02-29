# Time series forecasting with Legendre Memory Units (LMU)

An implementation of the Legendre Memory Unit (LMU) in jax/flax for time series forecasting. The LMU is a novel memory cell that can be used in recurrent neural networks (RNNs) to process time series data. The LMU is designed to capture long-range dependencies in time series data, and is particularly well-suited for time series forecasting tasks.

The original paper at NeurIPS 2019 can be found [here](https://proceedings.neurips.cc/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf)

## The data

We use the electric transform temperature prediction dataset from [here](https://github.com/zhouhaoyi/ETDataset).


## The model

We implement the Legendre Memory Unit (LMU) in jax/flax as described in the original paper. We compare it with the LSTM models provided by the flax library.