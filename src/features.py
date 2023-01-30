from typing import Tuple

import numpy as np
import pandas as pd


def create_trade_features(
    ob_df: pd.DataFrame,
    trades_df: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Creates trade features based on order book and trades data.

    Args:
        ob_df (pd.DataFrame): Order book data.
        trades_df (pd.DataFrame): Trades data.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Tuple with DataFrame containing order book data with additional trade features
            and array with order book ts data.
    """
    trades_ts = trades_df["TS"]
    ob_ts = ob_df["TS"]
    last_trade_idx = trades_ts.searchsorted(ob_ts, side='right') - 1
    last_trade_idx[0] = 0

    last_trade = trades_df.iloc[last_trade_idx].reset_index(drop=True)
    last_price_diff = (
        last_trade["Price"]
                  .diff()
                  .rename("last_price_diff")
                  .fillna(0)
    )
    last_amount = last_trade["Amount"].rename("last_amount")

    df = pd.concat([ob_df, last_amount, last_price_diff], axis=1).drop(columns="TS")
    ts = ob_ts.to_numpy()

    return df, ts

def create_time_insensitive(df: pd.DataFrame) -> pd.DataFrame:
    """Creates time-insensitive features based on order book data.

    Args:
        df (pd.DataFrame): Order book data.

    Returns:
        pd.DataFrame: A DataFrame with time-insensitive features added.
    """
    ask = [f"ask_{i}" for i in range(1, 6)]
    askv = [f"askv_{i}" for i in range(1, 6)]
    bid = [f"bid_{i}" for i in range(1, 6)]
    bidv = [f"bidv_{i}" for i in range(1, 6)]

    # Price features
    df["mid_price_1"] = (df["ask_1"] + df["bid_1"]) / 2

    df["ask_5_diff_ask_1"] = df["ask_5"] - df["ask_1"]
    df["bid_1_diff_bid_5"] = df["bid_1"] - df["bid_5"]

    df["ask_2_diff_ask_1"] = np.abs(df["ask_2"] - df["ask_1"])
    df["bid_2_diff_bid_1"] = np.abs(df["bid_2"] - df["bid_1"])

    # Volume features
    df["total_volume_diff"] = df[askv].sum(axis=1) - df[bidv].sum(axis=1)

    df['bidv_2_to_5'] = df[[f'bidv_{i}' for i in range(2, 6)]].sum(axis=1)
    df['askv_2_to_5'] = df[[f'askv_{i}' for i in range(2, 6)]].sum(axis=1)

    df["askv_1_ratio"] = df["askv_1"] / df[askv].sum(axis=1)
    df["bidv_1_ratio"] = df["bidv_1"] / df[bidv].sum(axis=1)

    # Drop unnecessary features
    df = df.drop(columns=ask[1:] + bid[1:] + askv[4:] + bidv[4:])

    return df

def create_time_sensitive(df: pd.DataFrame) -> pd.DataFrame:
    """Creates time-sensitive features based on order book data

    Args:
        df (pd.DataFrame): Order book data.

    Returns:
        pd.DataFrame: A DataFrame with time-sensitive features added.
    """
    df["ask_1_return"] = df["ask_1"].pct_change()
    df["bid_1_return"] = df["bid_1"].pct_change()

    for period in [30, 60, 100]:
        df[f"mid_price_1_return_{period}"] = df["mid_price_1"].pct_change(periods=period)

    df = df.drop(columns=["mid_price_1", "ask_1", "bid_1"])

    return df