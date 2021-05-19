import os
from argparse import ArgumentParser

import investpy
import numpy as np
import pandas as pd

# import requests
from tqdm import tqdm

from data_utils import get_total_df_from_candles, read_all_candles

# import yfinance as yf


def download(symbol, save_dir):
    # data = yf.download(symbol, start_date, end_date)
    # return data
    df = investpy.get_stock_historical_data(
        stock=symbol,
        country="United States",
        from_date="18/12/2018",
        to_date="20/02/2021",
    )
    df.drop("Currency", axis=1, inplace=True)
    # for col in df.columns:
    #    df.loc[:, col] = np.log(df.loc[:, col]) - np.log(df.loc[:, col].shift(1))
    # df = df.iloc[1:, :]
    # df = df.resample('H').pad()
    df.to_csv(f"{save_dir}/{symbol}.csv", index=True, header=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_file",
        required=False,
        default="s-and-p-500-companies-financials/data/constituents.csv",
        type=str,
        help="Data for the symbols to download.",
    )
    parser.add_argument(
        "--save_dir",
        required=False,
        default="stocks_raw2",
        type=str,
        help="Directory to save the downloaded data.",
    )
    args = parser.parse_args()
    # candles = read_all_candles()
    # df = get_total_df_from_candles(candles)
    # submission = pd.read_csv("submission/submission.csv", parse_dates=True, index_col="eod_ts")
    os.makedirs(args.save_dir, exist_ok=True)
    companies = pd.read_csv(args.data_file)
    try:
        symbols = companies["symbol"]
    except Exception as e:
        print(e)
        symbols = companies["Symbol"]
    # symbols_nasdaq = pd.read_csv("nasdaq_listings/nasdaq_listings.csv.save")["Symbol"]
    # total_symbols = symbols.tolist() + symbols_nasdaq.tolist()
    for symbol in tqdm(symbols, desc="Downloading assets data"):
        try:
            download(symbol, args.save_dir)
        except Exception as e:
            print(e)
            continue
