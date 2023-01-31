from typing import Tuple

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


def _split_data(df: pd.DataFrame) -> Tuple[Pool, Pool]:
    """Splits the given data into training and validation sets.

    Args:
        df (pd.DataFrame): DataFrame containing features and target data.

    Returns:
        Tuple[Pool, Pool]: A tuple of two `Pool` objects, one for training data and one for validation data.
    """        
    x = df.drop(columns=["returns"])
    y = df["returns"]

    x_train, y_train = x[:-150_000], y.iloc[:-150_000]
    x_val,   y_val   = x[-100_000:], y.iloc[-100_000:]

    train_pool = Pool(x_train, y_train)
    val_pool = Pool(x_val, y_val)

    return train_pool, val_pool

def fit(df: pd.DataFrame) -> catboost.core.CatBoostRegressor:
    """Fits a CatBoostRegressor model to training data.

    Args:
        df (pd.DataFrame): DataFrame containing features and target data.

    Returns:
        catboost.core.CatBoostRegressor: Fitted CatBoostRegressor model.

    """
    train_pool, val_pool = _split_data(df)
    
    model = CatBoostRegressor(
        iterations=1500, 
        learning_rate=0.07,
        random_seed=42, 
        eval_metric="R2"
    )
    model.fit(
        train_pool, 
        eval_set=val_pool, 
        early_stopping_rounds=100, 
        verbose=100
    )
    
    return model

def save_model(model: catboost.core.CatBoostRegressor) -> None:
    """Saves the model to disk.

    Args:
        model (catboost.core.CatBoostRegressor): CatBoostRegressor object to be saved.
    """
    model.save_model("return-prediction/models/catboost_model")

def load_model() -> catboost.core.CatBoostRegressor:
    """Loads saved model from disk.

    Returns:
        catboost.core.CatBoostRegressor: Trained and saved CatBoostRegressor model.
    """
    model = CatBoostRegressor()
    model.load_model("./return-prediction/models/catboost_model")
    
    return model

def forecast(df: pd.DataFrame, model: catboost.core.CatBoostRegressor) -> np.ndarray:
    """Forecasts using trained CatBoostRegressor model.

    Args:
        df: pd.DataFrame - A DataFrame containing the feature data to make predictions on.
    model: catboost.core.CatBoostRegressor - A trained CatBoostRegressor model.

    Returns:
    np.ndarray - An array of predictions made by the model.
    """
    
    y_preds = model.predict(df)
    
    return y_preds