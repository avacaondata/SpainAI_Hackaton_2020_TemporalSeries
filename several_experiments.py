import os
from argparse import ArgumentParser

import pandas as pd
import torch
from tqdm import tqdm

from data_utils import (_cut_special, create_stocks_df,
                        create_synthetic_future_df, cut_df, filter_assets,
                        get_total_df_from_candles, read_all_candles)
from deepdow.callbacks import EarlyStoppingCallback, MLFlowCallback
from opt_nets import (EconomistNet, MyNetwork, ResampleNetwork, SoftMaxNetwork,
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

params = {
    "lookback": [24, 15, 30],
    "horizon": [24, 15, 30],
    "lr": [1e-4, 1e-3, 1e-2],
    "epochs": [10, 20],
}

assets_0 = [
    'YAX',
    'OOS',
    'USX',
    'FSK',
    'TMF',
    'TDD',
    'YEC',
    'ZAB',
    'MUF',
    'PUL',
    'LWE',
    'REU',
    'ACY',
    'MMY',
    'ZXW',
    'WFJ',
    'PEW',
    'LWK',
    'ULI',
    'AUX',
    'HCC',
    'JNE',
    'JTL',
    'UEI',
    'THA',
    'TKT',
    'SKN',
    'YFC',
    'GGR',
    'NYP',
    'MET',
    'SYO',
    'BZC',
    'OXR',
    'BSX',
    'PME',
    'FNM',
    'EEY',
    'WWT',
    'TXR',
    'RAT',
    'RWJ'
 ]

assets_0_1 = [
    'ZVQ',
    'NCT',
    'OOS',
    'CSB',
    'UYZ',
    'TRO',
    'ERO',
    'AWW',
    'ACY',
    'MMY',
    'LUG',
    'LWK',
    'ZCD',
    'LHB',
    'NYD',
    'BFS',
    'SKN',
    'GGR',
    'TER',
    'NYP',
    'EOP',
    'PME',
    'FNM',
    'EEY',
    'ERQ',
    'AZG',
    'OJG',
    'WWT',
    'BOT',
    'TXR',
    'DIG',
    'PHI'
 ]

def save_network(experiment_name, run):
    network = run.models["main"]
    torch.save(network.state_dict(), f"models_2203/{experiment_name}.pt")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lookback", type=int, required=False, default=24, help="Number of timesteps to look back")
    parser.add_argument("--gap", default=1, type=int, required=False, help="Number of timesteps to leave before predicting")
    parser.add_argument("--horizon", default=24, type=int, required=False, help="Number of timesteps to predict ahead.")
    parser.add_argument("--optimizer", required=False, default="adam", type=str, help="Optimizer to use")
    parser.add_argument("--lr", required=False, default=0.0001, type=float, help="Learning rate to use.")
    parser.add_argument("--run_name", required=False, default="nconet", type=str, help="The number to put to the run.")
    parser.add_argument("--epochs", required=False, default=10, type=int)
    parser.add_argument("--save_name", default="submission_2103.csv", type=str, required=False, help="Name to put to submission")
    args = parser.parse_args()
    os.makedirs("mlflow_runs", exist_ok=True)
    os.makedirs("models_2203", exist_ok=True)
    candles = read_all_candles()
    candles1 = {k: v for k, v in candles.items() if k not in assets_0_1}
    candles2 = {k: v for k, v in candles.items() if k not in assets_0}
    df = get_total_df_from_candles(candles1, add_random=False)
    df2 = get_total_df_from_candles(candles2, add_random=False)
    df2 = _cut_special(df2, first="2020-02-01")
    n_timesteps, n_channels, n_assets, asset_names = get_variables_for_training(df2)
    train_dataloader, val_dataloaders, dataset = get_dataloaders(df,  args.lookback, args.gap, args.horizon)
    train_dataloader2, val_dataloaders2, dataset2 = get_dataloaders(df2,  args.lookback, args.gap, args.horizon)
    experiments = [
        # {
        #     "network": MyNetwork(
        #         n_assets,
        #         max_weight=0.2,
        #         force_symmetric=True,
        #         n_clusters=5,
        #         n_init=100,
        #         init="k-means++",
        #         random_state=None),
        #     "train_data": train_dataloader,
        #     "val_data": val_dataloaders,
        #     "name": "mynet_data1"
        # },
        {
            "network": MyNetwork(
                n_assets,
                max_weight=0.2,
                force_symmetric=True,
                n_clusters=5,
                n_init=100,
                init="k-means++",
                random_state=None),
            "train_data": train_dataloader2,
            "val_data": val_dataloaders2,
            "name": "mynet_data2"
        },
        # {
        #     "network": SoftMaxNetwork(
        #         n_assets,
        #         max_weight=0.2,
        #         force_symmetric=True
        #     ),
        #     "train_data": train_dataloader,
        #     "val_data": val_dataloaders,
        #     "name": "softmaxnet1"
        # },
        # {
        #     "network": SoftMaxNetwork(
        #         n_assets,
        #         max_weight=0.2,
        #         force_symmetric=True
        #     ),
        #     "train_data": train_dataloader2,
        #     "val_data": val_dataloaders2,
        #     "name": "softmaxnet2"
        # },
    ]
    for experiment in tqdm(experiments, desc="performing experiments..."):
        network = experiment["network"]
        run = train_model(
            network,
            experiment["train_data"],
            experiment["val_data"],
            create_optimizer(network, args.optimizer, args.lr),
            callbacks=[
                MLFlowCallback(
                    args.run_name,
                    mlflow_path="./mlflow_runs",
                    experiment_name=experiment["name"],
                    log_benchmarks=True
                ),
                EarlyStoppingCallback("val", "loss")
            ],
            epochs=args.epochs
        )
        save_network(experiment["name"], run)
