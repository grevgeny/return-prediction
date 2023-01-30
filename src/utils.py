from typing import Tuple

import h5py
import numpy as np
import pandas as pd


def read_data_h5_file(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads data from data HDF5 file and returns the order book and trades data.

    Args:
        path (str): The path to the HDF5 file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of two pandas DataFrames, one for the order book data and one for the trades data.
    """
    OB_dfs = []
    trades_dfs = []

    with h5py.File(path, "r") as f:
        for group in ["OB", "Trades"]:
            for name, data in f[group].items():
                if name in ["TS", "Amount", "Price"]:
                    data_df = pd.DataFrame(data[()], columns=[name])
                else:
                    data_df = pd.DataFrame(
                        data[:, :5],
                        columns=[f"{name.lower()}_{i}" for i in range(1, 6)],
                        dtype="float32"
                    )
                
                if group == "OB":
                    OB_dfs.append(data_df)
                else:
                    trades_dfs.append(data_df)

    OB_df = pd.concat(OB_dfs, axis=1)
    trades_df = pd.concat(trades_dfs, axis=1)

    return OB_df, trades_df


def read_result_h5_file(path: str) -> pd.DataFrame:
    """Reads data from HDF5 file and returns mid-price returns data.
    
    Args:
        path (str): The path to the HDF5 file.
    
    Returns:
        np.ndarray: A numpy array containing the result data from the file.
    """
    with h5py.File(path, "r") as f:
        returns = f["Return/Res"][()]
    
    return returns


def save_results(y_preds: np.ndarray, TS: np.ndarray) -> None:
    """Saves forecasting results to an HDF5 file.

    Args:
        y_preds (np.ndarray): Array with predicted returns.
        TS (np.ndarray): Array with time index.
    """
    with h5py.File("./results/forecast.h5", "w") as f:
        group = f.create_group("Return")
        group.create_dataset(name='TS', data=TS)
        group.create_dataset(name='Res', data=y_preds)