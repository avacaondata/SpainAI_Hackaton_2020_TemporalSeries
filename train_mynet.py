import os
from argparse import ArgumentParser

import torch

from data_utils import get_total_df_from_candles, read_all_candles
from deepdow.callbacks import EarlyStoppingCallback, MLFlowCallback
from opt_nets import (MyNetwork, NumericalRiskNetwork, ResampleNetwork,
                      get_dataloaders, get_predictions,
                      get_variables_for_training, train_model)
from submission_utils import (_weights_to_dict, general_weights_fixer,
                              get_submission_markowitz, test_submission)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def create_optimizer(network, opt_type, lr):
    if opt_type == "adam":
        return torch.optim.Adam(network.parameters(), lr)
    else:
        return None

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
        default=60,
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
        default=60,
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
        default="nconet",
        type=str,
        help="The number to put to the run.",
    )
    parser.add_argument("--epochs", required=False, default=100, type=int)
    parser.add_argument(
        "--save_name",
        default="submission_2503.csv",
        type=str,
        required=False,
        help="Name to put to submission",
    )
    parser.add_argument(
        "--save_dir",
        default="models_2503",
        type=str
    )

    args = parser.parse_args()
    os.makedirs("mlflow_runs", exist_ok=True)
    candles = read_all_candles()
    df = get_total_df_from_candles(candles, add_random=False)
    n_timesteps, n_channels, n_assets, asset_names = get_variables_for_training(df)
    train_dataloader, val_dataloaders, dataset = get_dataloaders(
        df, args.lookback, args.gap, args.horizon
    )
    #network = MyNetwork(
    #    n_assets,
    #    max_weight=0.2,
    #    force_symmetric=True,
    #    n_clusters=5,
    #    n_init=100,
    #    init="k-means++",
    #    random_state=None,
    #)  # ResampleNetwork(n_assets)

    network = NumericalRiskNetwork(
            n_assets,
            max_weight=0.2,
            force_symmetric=True
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
                experiment_name="nconet",
                log_benchmarks=True,
            ),
            EarlyStoppingCallback("val", "loss"),
        ],
        epochs=args.epochs,
        loss_="maximum_drawdown"
    )
    save_network("riskbudget", run, direc=args.save_dir)
    weights_predicted = get_predictions(run, n_channels, args.lookback, n_assets)
    weights = general_weights_fixer(weights_predicted)
    weights = _weights_to_dict(weights=weights, asset_names=asset_names)
    submission = get_submission_markowitz(weights, asset_names)
    submission.to_csv(args.save_name, index=True, header=True)
    test_submission(submission)
