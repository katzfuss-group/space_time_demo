# Generative spatio-temporal autoregressive Gaussian processes

## About this project

This is the repository that applies the transport map model (also known as the auto-regressive GP model) from [Matthias' paper](https://www.tandfonline.com/doi/full/10.1080/01621459.2023.2197158), developed for spatial data, to spatio-temporal data. A toy example with a precipitation dataset is provided.

## Install dependencies
It is suggested that you create a new virtual environment to install these dependencies, for example, with `python3 -m venv .venv` and run `source .venv/bin/activate`.
```bash
pip install ./dependent_packages/NN_L2/
pip install torch
pip install --no-build-isolation ./dependent_packages/maxmin_exact/
pip install ./dependent_packages/batram/
pip install -r requirements.txt
```

## Run the toy example
```bash
python3 main.py
```