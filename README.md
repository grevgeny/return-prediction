# Return Prediction for Cryptoasset

This repository contains code for a machine learning model that predicts mid return for cryptoasset. The model is based on CatBoostRegressor and utilizes data from order book (OB) and trades of cryptoasset.

## Repository Structure
```
README.md
setup.py
data/
    train/
        data.h5
        result.h5
src/
    data_processing.py
    features.py
    models.py
    run_forecasting.py
    utils.py
models/
    catboost_model
results/
```
The repository contains the following directories:

- `data/train`: Directory with training data, data.h5 and result.h5.
- `src/`: Directory with source code.
- `models/`: Directory with saved model.
- `results/`: Directory with results of forecasting.

## Installation
In this project Python 3.10.6 has been used.

```bash
git clone https://github.com/grevgeny/return-prediction.git;
pip install . --quiet
```

## Usage
The script can be run using the run_forecasting command. The mode of operation and path to data must be specified:
```
run_forecasting --mode <mode> --data_path <path>
```
where:

- `mode`: Specifies mode of operation, either __fit__ or __forecast__.
- `data_path`: Path to data directory with data.h5 and result.h5 (__fit__) or to data.h5 directly (__forecast__).

## Examples
Assuming following data folder structure:  
```
data/  
    train/
        data.h5
        result.h5
    test/
        data.h5
```

To fit the model, run the following command:
```
run_forecasting --mode fit --data_path data/train/
```
It fits the model and saves it to `./model/catboost_model`.

To forecast, run the following command:
```
run_forecasting --mode forecast --data_path data/test/data.h5
```
It uses trained model at `./model/catboost_model` and produces forecast to `./results/forecast.h5`.