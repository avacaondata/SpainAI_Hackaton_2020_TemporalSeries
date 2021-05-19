import os
import pickle
import random
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm, trange

from constants_variables import logret_cols


def _save_pickle(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def _load_pickle(file):
    with open(file, "rb") as f:
        pickle.load(f)


def read_all_candles(
    folder="/home/alejandro.vaca/reto_series_temporales/trainTimeSeries/trainTimeSeries/TrainCandles",
):
    """
    Reads candles from the folder and stores them in a dictionary
    where keys are asset names and values are the candles dfs.

    Parameters
    ----------
    folder: str
        Where data is located.

    Returns
    -------
    data: Dict
        dictionary with data descr. above.
    """
    data = {}
    for file in tqdm(os.listdir(folder), desc="Getting candles data for Darwins..."):
        df = pd.read_csv(f"{folder}/{file}")
        df.columns = ["date", "close", "max", "min", "open"]
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        # idx = pd.DatetimeIndex(df["date"], freq="H")
        # df.drop("date", axis=1, inplace=True)
        # df.set_index(idx, inplace=True)
        name = file.replace("DARWINUniverseCandlesOHLC_", "").replace("_train.csv", "")
        data[name] = df
    return data


def compute_returns(v, remove_others=False):
    """Computes returns from a dataframe"""
    v["returns"] = (v["close"] - v["open"]) / v["open"]
    if remove_others:
        v.drop([col for col in v.columns if col != "returns"], axis=1, inplace=True)
    return v


def filter_candles(candles, remove):
    """Removes elements in param remove from candles."""
    return {k: v for k, v in candles.items() if k not in remove}


def _add_random_column(df):
    """
    Useful when we're having issues due to mismatch of channels
    btw. assets and external vars.
    """
    df["randomcol"] = [random.randint(100, 400)] * df.shape[0]
    return df


def get_total_df_from_candles(candles, remove=None, add_random=False):
    """Gets a dataframe with all the assets in candles."""
    if remove is not None:
        candles = filter_candles(candles, remove)
    if add_random:
        for asset in candles:
            candles[asset] = _add_random_column(candles[asset])
    df = pd.concat(candles.values(), keys=candles.keys(), axis=1)
    df = df.resample("H").aggregate("mean")
    return df


def get_returns_and_cov(df):
    """Gets returns and cov matrix from assets df with only close vals."""
    cov = df.cov().values
    exp_returns = df.mean().values.reshape(-1, 1)
    return cov, exp_returns


def read_all_scores(folder="trainTimeSeries/trainTimeSeries/TrainScores"):
    """
    Reads scores from the folder.
    """
    scores = {}
    for file in tqdm(os.listdir(folder)):
        df = pd.read_csv(f"{folder}/{file}", parse_dates=True, index_col="eod_ts")
        name = file.replace("scoresData_", "").replace("_train.csv", "")
        scores[name] = df
    return scores


def group_dates_df(
    df,
    period="D",
    mincol="low",
    opencol="open",
    highcol="high",
    closecol="close",
    volcol="volume",
):
    """
    Groups dates_df by period, default is daily.
    Then, creates new df with OHLC data.
    """
    df = df.resample(period).aggregate(["first", "last", "min", "max", "mean"])
    newdf = pd.DataFrame({"date": df.index}).set_index("date")
    newdf["close"] = df[closecol, "last"].values
    newdf["low"] = df[mincol, "min"].values
    newdf["high"] = df[highcol, "max"].values
    newdf["open"] = df[opencol, "first"].values
    if volcol in df.columns:
        newdf["volume"] = df[volcol, "mean"].values
    return newdf


def create_prices_df(
    dfs_dict, exact=True, delta=pd.Timedelta("5m"), col_="close", select_one_col=True
):
    """
    Merges all prices dataframes into a single one. It can be exact
    (all dates are included even if some assets don't have values
    for those dates) or approximate, which means that dates almost
    equal by less than delta, are treated as equal.
    """
    dfs = []
    for asset in tqdm(dfs_dict):
        dfs_dict[asset].columns = [
            f"{asset}_{col}" if col != "date" else col
            for col in dfs_dict[asset].columns
        ]
        if select_one_col:
            dfs.append(dfs_dict[asset][f"{asset}_{col_}"])
        else:
            dfs.append(dfs_dict[asset])
    if exact:
        return reduce(
            lambda left, right: pd.merge(left, right, how="outer", on="date"), dfs
        )
    else:
        return reduce(
            lambda left, right: pd.merge_asof(
                left,
                right,
                on="date",
                tolerance=delta,
            ),
            dfs,
        )


def get_prices_daily(folder):
    """Get a df of all assets aggregated daily."""
    candles = read_all_candles(folder)
    candles = {
        k: group_dates_df(v)
        for k, v in tqdm(candles.items(), desc="Creating groups per hour")
    }
    return create_prices_df(candles)


def get_raw_dataset(folder):
    """
    Gets dataset of all assets in hourly format.
    Also fills NAs with last available value.
    """
    candles = read_all_candles(folder)
    df = create_prices_df(candles)
    df = df.resample("H").aggregate("mean")
    df.fillna(method="ffill", inplace=True)
    return df


def _read_sp500(
    file="/home/alejandro.vaca/reto_series_temporales/datos_series_temporales/IVE_tickbidask.txt",
):
    """Reads sp500 index file, parsing dates."""
    df = pd.read_csv(
        file,
        names=["date", "time", "price", "bid", "ask", "volume"],
        parse_dates=[["date", "time"]],
    )
    df.set_index("date_time", inplace=True)
    return df


def _read_oil(
    file="/home/alejandro.vaca/reto_series_temporales/datos_series_temporales/OIH_adjusted.txt",
):
    df = pd.read_csv(
        file,
        names=["date", "time", "open", "high", "low", "close", "volume"],
        parse_dates=[["date", "time"]],
    )
    df.set_index("date_time", inplace=True)
    return df


def _read_treasury(
    file="/home/alejandro.vaca/reto_series_temporales/datos_series_temporales/10-year-treasury-bond-rate-yield-chart.csv",
):
    df = pd.read_csv(file, skiprows=15, parse_dates=["date"], index_col="date")
    return df


def _read_all_other(folder="/home/alejandro.vaca/reto_series_temporales/FOREX/others/", savedir="alternative_stocks"):
    files = [file for file in os.listdir(folder) if ".csv" in file]
    for file in tqdm(files, desc="Iterating over others..."):
        df = _read_other_df(folder + file)
        df = _cut_special(df, first="2018-12-01", last="2021-01-20")
        df.index.rename("Date", inplace=True)
        df.to_csv(f"{savedir}/{file}", header=True, index=True)


def _read_other_df(df_name):
    df = pd.read_csv(df_name, sep="\t", names=["Date", "Open", "High", "Low", "Close", "Volume"], parse_dates=True, index_col="Date")
    # df.set_index("Date", inplace=True)
    df = group_dates_df(df, mincol="Low", opencol="Open", highcol="High", closecol="Close", volcol="Volume")
    return df



def _read_forex_df(df_name):
    """
    data structure is:
    <TICKER>,<DTYYYYMMDD>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>
    """
    df = pd.read_csv(df_name, parse_dates=["<DTYYYYMMDD>"], index_col="<DTYYYYMMDD>")
    df = df[[col for col in df.columns if col not in ["<TICKER>", "<TIME>"]]]
    df.columns = [col.replace("<", "").replace(">", "").lower() for col in df.columns]
    df = group_dates_df(df, volcol="vol")
    df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "vol": "Volume",
        },
        inplace=True,
    )
    df.index.rename("Date", inplace=True)
    # df = df.resample("D").aggregate("mean")
    return df


def _read_all_forex(folder="/home/alejandro.vaca/reto_series_temporales/FOREX/", savedir="alternative_stocks"):
    files = [file for file in os.listdir(folder) if ".txt" in file]
    for file in tqdm(files, desc="Iterating over FOREX files"):
        df = _read_forex_df(folder + file)
        df = _cut_special(df, first="2018-12-01", last="2021-01-20")
        df.to_csv(f"{savedir}/{file.replace('.txt', '.csv')}")


def cut_df(
    df, assets_df, prefix=None, until_test=False, last_date="2020-12-24 21:00:00"
):
    """Cuts a dataframe based on a reference assets_df."""
    first_date = assets_df.index[0]
    after_date = df.index >= first_date
    if until_test:
        before_date = df.index <= pd.to_datetime(last_date)
    else:
        last_date = assets_df.index[-1]
        before_date = df.index <= last_date
    between = after_date & before_date
    df_cutted = df.loc[between]
    if prefix is not None:
        df_cutted = _fix_df(df_cutted, prefix)
    return df_cutted


def _cut_special(df, first="2018-12-19", last="2020-12-24"):
    first = pd.to_datetime(first)
    last = pd.to_datetime(last)
    after_date = df.index >= first
    before_date = df.index <= last
    df_cutted = df.loc[after_date & before_date]
    return df_cutted


def _fix_df(df, prefix, period="H"):
    df.columns = ["date"] + [
        f"{prefix}_{col}" for col in df.columns if "date" not in col
    ]
    df = df.resample(period, on="date").aggregate("mean")
    df.fillna(method="ffill", inplace=True)
    return df


#   NOTE: SUBSTITUTED BY CUT_DF


def cut_sp500(df_sp, assetsdf):
    """Cuts sp500 for dates that only appear in assetsdf."""
    first_date = assetsdf.index[0]
    last_date = assetsdf.index[-1]
    after_first = df_sp["date_time"] >= first_date
    before_last = df_sp["date_time"] <= last_date
    between = after_first & before_last
    df = df_sp.loc[between]
    df = _fix_df(df, "sp500")
    return df


def cut_oil(oil_df, assetsdf):
    """Cuts oil df for relevant dates"""
    first_date = assetsdf.index[0]
    last_date = assetsdf.index[-1]
    after_first = oil_df["date_time"] >= first_date
    before_last = oil_df["date_time"] <= last_date
    between = after_first & before_last
    oil_df = oil_df.loc[between]
    oil_df = _fix_df(oil_df, "oil")
    return oil_df


def _save_scaler_variable(df, col, name):
    arr = df[col].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(arr)
    _save_pickle(scaler, f"scaler_{name}.pkl")
    return scaler


def preprocess_data(df, reference_df, name, scaler):
    df = _fill_dates(df, reference_df)
    if f"{name}_bid" in df.columns:
        df[f"{name}_spread"] = (df[f"{name}_ask"] - df[f"{name}_bid"]) / df[
            f"{name}_price"
        ]
    cols_logret = [col for col in df.columns if any([c in col for c in logret_cols])]
    for col in cols_logret:
        df.loc[:, col] = np.log(df[col]) - np.log(df[col].shift(1))
    df[f"{name}_volume"] = scaler.transform(df[f"{name}_volume"].values.reshape(-1, 1))
    return df


def _postprocess(df):
    df.fillna(0, inplace=True)
    return df


def _fill_dates(df, ref_df):
    dates_ = pd.DataFrame({"date": ref_df.index}).set_index("date")
    df = pd.merge(df, dates_, left_index=True, right_index=True, how="outer")
    df.fillna(method="ffill", inplace=True)
    return df


"""
def _get_time_attributes(df):
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    sin_day, cos_day = _get_sine_cosine(df.day, 30)
    sin_hour, cos_hour = _get_sine_cosine(df.hour, 24)
    df.drop(["day", "hour"], axis=1, inplace=True)
    df["sin_day"] = sin_day
    df["cos_day"] = cos_day
    df["sin_hour"] = sin_hour
    sin = np.sin(
        2 * np.pi * dates / period
    )
    cos = np.cos(
        2 * np.pi * dates / period
    )
    return sin, cos


def build_obs_df(df, assetsdf):
    df = _get_time_attributes(df)
    df = _which_assets_available(df, assetsdf)
    return df
"""


def _which_assets_available(df, assets_df, test=False, test_shape=None):
    assets_df.fillna(method="ffill", inplace=True)
    if not test:
        values = [
            assets_df.iloc[i, :].notna().values.astype("int")
            for i in trange(assets_df.shape[0])
        ]
    else:
        values = [[1] * assets_df.shape[1]] * test_shape
    columns = [f"{col.replace('_close', '')}_available" for col in assets_df.columns]
    df_available = pd.DataFrame(values, columns=columns, index=assets_df.index)
    df = pd.merge(df, df_available, left_index=True, right_index=True, how="outer")
    return df


def create_stocks_df(
    stocks_dir="/home/alejandro.vaca/reto_series_temporales/alternative_stocks",
    aggregate_hourly=False,
    filt=None,
):
    """Creates a dataframe with all the stocks we've got data from."""
    stocks = [stock.replace(".csv", "") for stock in os.listdir(stocks_dir)]
    # if filt is not None:
    #    stocks = filter_stocks(stocks)
    stocks_dict = {
        stock: pd.read_csv(
            f"{stocks_dir}/{stock}.csv", index_col="Date", parse_dates=True
        )
        for stock in tqdm(stocks, desc="Loading stocks.")
    }
    for stockname in stocks_dict:
        stocks_dict[stockname].columns = [
            col.lower() for col in stocks_dict[stockname].columns
        ]
    stocks_df = pd.concat(stocks_dict.values(), keys=stocks_dict.keys(), axis=1)
    # stocks_df = stocks_df.resample("H").aggregate("mean")
    return stocks_df


def create_synthetic_future_df(candles, submission_df):
    dates = submission_df.index
    # columns = [col for col in assets_df.columns]
    dates_dict = {"date": dates}
    candles_synthetic = {
        k: pd.DataFrame(
            {
                **dates_dict,
                **{
                    col: [0.0] * len(dates)
                    for col in ["close", "max", "min", "open", "randomcol"]
                },
            }
        ).set_index("date")
        for k in candles
    }
    df = pd.concat(candles_synthetic.values(), keys=candles_synthetic.keys(), axis=1)
    # d_ = {col: [0]*len(dates) for col in columns}
    # df = pd.DataFrame({"date": dates, **d_}).set_index("date")
    return df


def _compute_return_volatility(df):
    df2 = df.dropna()
    returns = df2["close"].pct_change()[1:]
    var = returns.std()
    ret = returns.mean()
    return ret, var


def _compute_return_pre_post_covid(df):
    precovid = df.index <= pd.to_datetime("2020-03-01")
    df_precovid = df.loc[precovid]
    df_postcovid = df.loc[~precovid]
    return_precovid = df_precovid["close"].pct_change()[1:].mean()
    return_postcovid = df_postcovid["close"].pct_change()[1:].mean()
    return return_precovid, return_postcovid


def _prepare_scores_df(df):
    df.drop(["fcal_ts", "lcal_ts"], axis=1, inplace=True)
    # df.set_index("eod_ts", inplace=True)
    df["mean_score"] = df.mean(axis=1).values
    return df


def _compute_weighted_mean(df):
    means = df["Pf"].values  # mean_score
    weights = np.linspace(0.0, 1.0, len(means))
    mean = sum([mu * w / sum(weights) for mu, w in zip(means, weights)]) / len(means)
    return mean


def _filter_by_scores(scores):
    # mean_score = df["mean_score"].mean()
    scores_mean = {
        k: _compute_weighted_mean(v) for k, v in scores.items()
    }  # v["mean_score"].ewm(halflife=5).mean().mean()
    remove_scores = []
    perc_20_score = np.percentile([v for v in scores_mean.values()], 30)  # 30
    for asset in scores:
        if scores_mean[asset] <= perc_20_score:
            remove_scores.append(asset)
    return remove_scores


def filter_assets(assets):
    """Returns list of assets that shouldn't be included in the wallet."""
    asset_names = [k for k in assets.keys()]
    scores = read_all_scores()
    scores = {k: _prepare_scores_df(v) for k, v in scores.items()}
    stats_dict = {
        asset: _compute_return_volatility(assets[asset]) for asset in asset_names
    }
    # all_rets = [t[0] for t in stats_dict.values()]
    # all_vars = [t[1] for t in stats_dict.values()]
    # perc_30_rets = np.percentile(all_rets, 5)
    # perc_70_vars = np.percentile(all_vars, 95)
    # remove = []
    # remove.extend([asset for asset in asset_names if asset not in scores])
    # for asset in stats_dict:
    #    if stats_dict[asset][0] < perc_30_rets:
    #        remove.append(asset)
    #    if stats_dict[asset][1] > perc_70_vars:
    #        remove.append(asset)
    # returns_pre_post = {
    #    asset: _compute_return_pre_post_covid(assets[asset])
    #    for asset in asset_names
    # }
    # diffs = [t[0]-t[1] for t in returns_pre_post.values()]
    # print(diffs)
    ## perc_80_diffs = np.percentile(diffs, 30)
    # for asset, diff in zip(returns_pre_post, diffs):
    #    if diff < 0:
    #        print("Appending")
    #        remove.append(asset)
    remove_scores = _filter_by_scores(scores)
    # remove.extend(remove_scores)
    # remove = remove + remove_scores
    return list(set(remove_scores))
