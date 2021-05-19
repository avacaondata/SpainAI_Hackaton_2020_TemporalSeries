import os
from argparse import ArgumentParser

import investpy
import pandas as pd
import requests
from tqdm import tqdm

# import yfinance as yf


def download(
    symbol,
    save_dir,
    start_date,
    end_date,
):
    # data = yf.download(symbol, start_date, end_date)
    # return data
    df = investpy.get_stock_historical_data(
        stock=symbol,
        country="United States",
        from_date="01/01/2010",
        to_date="01/01/2020",
    )
    df.drop("Currency", axis=1, inplace=True)
    df.to_csv(f"{save_dir}/symbol.csv", index=True, header=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_file", required=True, type=str, help="Data for the symbols to download."
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        type=str,
        help="Directory to save the downloaded data.",
    )
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    companies = pd.read_csv(args.data_file)
    try:
        symbols = companies["symbol"]
    except:
        symbols = companies["Symbol"]
    for symbol in symbols:
        download(symbol, args.save_dir)
