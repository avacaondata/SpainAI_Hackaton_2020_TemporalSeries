import os
from argparse import ArgumentParser

import pandas as pd
import requests
from tqdm import tqdm

from data_utils import (_cut_special, _read_all_forex, _read_all_other,
                        _read_oil, _read_sp500, _read_treasury, group_dates_df)


def process_crypto_data(symbols=["BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT"], savedir="alternative_stocks"):
    for symbol in tqdm(symbols, desc="Iterating over cryptos"):
        df = pd.read_csv(f"cryptodata/{symbol}-1d-data.csv")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df.rename(
            columns={
                "timestamp": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )
        df.set_index("Date", inplace=True)
        # df.index.rename("Date", inplace=True)
        df.to_csv(f"{savedir}/{symbol}.csv", header=True, index=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="alternative_stocks")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print("Processing others")
    _read_all_other()
    print("Processing forex...")
    _read_all_forex()
    print("Processing crypto...")
    process_crypto_data()
    # print("Processing oil...")
    # oil = _read_oil()
    # oil = group_dates_df(oil)
    # oil.rename(index = {"date_time": "Date", }, columns = {"open": "Open", "high": "High","low": "Low", "close": "Close", "volume": "Volume"}, inplace = True)
    # oil.index.rename("Date", inplace=True)
    # oil = _cut_special(oil, first="2018-12-01", last="2021-01-20")
    # oil.to_csv(f"{args.save_dir}/OIL.csv", header=True, index=True)
    # print("Processing sp500...")
    # sp500 = _read_sp500()
    # sp500 = group_dates_df(sp500, mincol="price", opencol="price", highcol="price", closecol="price", )
    # sp500.rename(index={"date_time": "Date", }, columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace = True)
    # s500 = _cut_special(sp500, first="2018-12-01", last="2021-01-20")
    # sp500.index.rename("Date", inplace=True)
    # sp500.to_csv(f"{args.save_dir}/sp500.csv", header=True, index=True)
    print("Processing interest rates")
    treasury = _read_treasury()
    treasury = group_dates_df(
        treasury,
        mincol="value",
        opencol="value",
        highcol="value",
        closecol="value",
        volcol="value",
    )
    treasury.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )
    treasury.index.rename("Date", inplace=True)
    treasury = _cut_special(treasury, first="2018-12-01", last="2021-01-20")
    treasury.to_csv(f"{args.save_dir}/treasury10yr.csv", header=True, index=True)
