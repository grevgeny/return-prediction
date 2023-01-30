from pathlib import Path
from typing import Tuple

import catboost
import pandas as pd
from catboost import CatBoostRegressor, Pool


def _split_data(X: pd.DataFrame, y: pd.Series) -> Tuple[Pool, Pool]:

    X_train, y_train = X[:-150_000], y.iloc[:-150_000]
    X_val,   y_val   = X[-100_000:], y.iloc[-100_000:]

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    return train_pool, val_pool

def fit(X: pd.DataFrame, y: pd.Series) -> catboost.core.CatBoostRegressor:
    train_pool, val_pool = _split_data(X, y)
    
    model = CatBoostRegressor(
        iterations=2000, 
        learning_rate=0.07,
        random_seed=42, 
        eval_metric="R2"
    )
    model.fit(
        train_pool, 
        eval_set=val_pool, 
        early_stopping_rounds=100, 
        verbose="Silent"
    )
    
    return model

def save_model(model: catboost.core.CatBoostRegressor) -> None:
    model.save_model("./model")

def load_model() -> catboost.core.CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model("/model")
    
    return model

def forecast(df: pd.DataFrame, model: catboost.core.CatBoostRegressor) -> None:
    ...
 

  