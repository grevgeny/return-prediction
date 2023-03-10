from typing import Tuple

import numpy as np
import pandas as pd

from .features import (create_time_insensitive, create_time_sensitive,
                      create_trade_features)
from .utils import read_data_h5_file, read_result_h5_file


def load_data(source: str, mode: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and returns the order book and trades data.
    Args:
        source (str): The source of the data, either a file path (forecast) or a directory (fit).
        mode (str): The mode to load the data, either "fit" or "forecast".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of two pandas DataFrames, one for the order book data and one for the trades data.
    """
    data_path = f"{source}/data.h5" if mode == "fit" else source
    OB_df, trades_df = read_data_h5_file(data_path)

    if mode == "fit":
        result_path = f"{source}/result.h5"
        returns_df = read_result_h5_file(result_path)
        OB_df["returns"] = returns_df

    return OB_df, trades_df

def process_data(OB_df: pd.DataFrame, trades_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Processes and creates features from order book and trades data.

    Args:
        OB_df (pd.DataFrame): DataFrame containing order book data.
        trades_df (pd.DataFrame): DataFrame containing trades data.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: A tuple of DataFrame containing processed and created features
            and array with TS data.

    """
    # Create fetures based on given trades occured
    df, TS = create_trade_features(OB_df, trades_df)

    # Create time-insensitive features based on LOB data
    df = create_time_insensitive(df)

    # Create time-sensitive features based on LOB data
    df = create_time_sensitive(df)

    return df, TS