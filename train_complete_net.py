import os
from argparse import ArgumentParser

import pandas as pd
import torch

from data_utils import (_cut_special, create_stocks_df,
                        create_synthetic_future_df, cut_df, filter_assets,
                        get_total_df_from_candles, read_all_candles)
from deepdow.callbacks import EarlyStoppingCallback, MLFlowCallback
from opt_nets import (EconomistNet, ResampleNetwork,
                      build_external_dataloaders, get_dataloaders,
                      get_predictions, get_test_dataloader,
                      get_variables_for_training, get_weights_submission,
                      train_model)
from submission_utils import (_convert_daily_submission, _weights_to_dict,
                              general_weights_fixer, get_submission_markowitz,
                              test_submission)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def create_optimizer(network, opt_type, lr):
    if opt_type == "adam":
        return torch.optim.Adam(network.parameters(), lr)
    else:
        return None


def _filter_submission(submission, first="2020-08-18", last="2020-12-24"):
    first = pd.to_datetime(first)
    last = pd.to_datetime(last)
    after_first = submission.index >= first
    before_last = submission.index <= last
    between = after_first & before_last
    return submission.loc[between]



def save_network(experiment_name, run, direc="models_2403"):
    os.makedirs(direc, exist_ok=True)
    network = run.models["main"]
    torch.save(network.state_dict(), f"{direc}/{experiment_name}.pt")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--lookback",
        type=int,
        required=False,
        default=24,
        help="Number of timesteps to look back",
    )
    parser.add_argument(
        "--gap",
        default=1,
        type=int,
        required=False,
        help="Number of timesteps to leave before predicting",
    )
    parser.add_argument(
        "--horizon",
        default=24,
        type=int,
        required=False,
        help="Number of timesteps to predict ahead.",
    )
    parser.add_argument(
        "--optimizer", required=False, default="adam", type=str, help="Optimizer to use"
    )
    parser.add_argument(
        "--lr", required=False, default=0.001, type=float, help="Learning rate to use."
    )
    parser.add_argument(
        "--run_name",
        required=False,
        default="resamplenet_experiment",
        type=str,
        help="The number to put to the run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=100,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--save_name",
        default="prueba.csv",
        type=str,
        required=False,
        help="Name to put to submission",
    )
    args = parser.parse_args()
    os.makedirs("mlflow_runs", exist_ok=True)
    candles = read_all_candles()
    submission_muestra = pd.read_csv(
        "submission/submission.csv", parse_dates=True, index_col="eod_ts"
    )
    # cols_ = [col.replace("allo_", "") for col in submission_muestra.columns]
    # candles = {k:v for k, v in candles.items() if k in cols_}
    # remove_assets = filter_assets(candles)
    # candles = {k: v for k, v in candles.items() if k not in remove_assets}
    df = get_total_df_from_candles(candles, add_random=True)
    df = df.resample("H").aggregate("mean")
    df_future_help = df.copy()
    # df = _cut_special(df, first="2020-02-01")
    stocks_df = create_stocks_df()
    stocks_df.fillna(method="ffill", inplace=True)
    stocks_df = _cut_special(stocks_df, first="2018-12-19", last="2020-06-17")

    # stocks_df = _cut_special(stocks_df, first="2020-02-01", last="2020-06-17")
    stocks_df.fillna(method="bfill", inplace=True)
    # stocks_df = cut_df(stocks_df, df,)
    n_timesteps, n_channels, n_assets, asset_names = get_variables_for_training(df)
    train_dataloader, val_dataloaders, dataset = build_external_dataloaders(
        stocks_df, df, args.lookback, args.gap, args.horizon, perc_val=0.25
    )

    network = EconomistNet(
        n_external=len(stocks_df.columns.levels[0]),
        n_assets=len(df.columns.levels[0]),
        max_weight=0.10,
    )
    run = train_model(
        network,
        train_dataloader,
        val_dataloaders,
        create_optimizer(network, args.optimizer, args.lr),
        callbacks=[
            MLFlowCallback(
                args.run_name,
                mlflow_path="./mlflow_runs",
                experiment_name="economistnet",
                log_benchmarks=True,
            ),
            EarlyStoppingCallback("val", "loss"),
        ],
        epochs=args.epochs,
        device="cuda",
        loss_="maximum_drawdown"
    )
    save_network("complete_2703", run, direc="models_2703")
    submission_df = pd.read_csv(
        "/home/alejandro.vaca/reto_series_temporales/submission/submission.csv",
        parse_dates=True,
        index_col="eod_ts",
    )
    # future_df = create_synthetic_future_df(candles, submission_df)
    stocks_test = create_stocks_df()
    stocks_test = stocks_test.resample("D").aggregate("mean").ffill()
    stocks_test = _cut_special(
        stocks_test, first="2020-05-01", last="2021-03-15"
    )  # "2020-07-25")
    stocks_hourly = stocks_test.resample("H").aggregate("mean")
    index = pd.date_range(start="2020-05-01", end="2021-03-15", freq="H")
    future_df = df_future_help.sample(stocks_hourly.shape[0]).copy().set_index(index)
    # n_timesteps, n_channels, n_assets, asset_names = get_variables_for_training(future_df)
    # future_df = future_df.resample("H").aggregate("mean")
    test_dataloader = get_test_dataloader(
        stocks_test, future_df, args.lookback, args.gap, args.horizon
    )
    submission = get_weights_submission(test_dataloader, run.models["main"])
    submission = _filter_submission(submission)
    submission.index = submission.index.rename("eod_ts")
    submission = _convert_daily_submission(submission, submission_muestra)
    submission.to_csv(args.save_name, index=True, header=True)
