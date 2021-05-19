import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import get_total_df_from_candles, read_all_candles
from deepdow.benchmarks import Benchmark, OneOverN
from deepdow.data import FlexibleDataLoader, InRAMDataset, RigidDataLoader
from deepdow.experiments import Run
from deepdow.layers import (NCO, RNN, AttentionCollapse, AverageCollapse, Conv,
                            Cov2Corr, CovarianceMatrix, NumericalMarkowitz,
                            NumericalRiskBudgeting, Resample, SoftmaxAllocator)
from deepdow.losses import MaximumDrawdown, SharpeRatio
from deepdow.nn import BachelierNet, KeynesNet, LinearNet, ThorpNet
from deepdow.utils import raw_to_Xy
from submission_utils import general_weights_fixer, get_submission_markowitz


def get_variables_for_training(df):
    """Uses df to return number timesteps, number of channels and number of assets."""
    raw_df = df.copy()
    n_timesteps = len(raw_df)  # 20
    n_channels = len(raw_df.columns.levels[1])  # 2
    n_assets = len(raw_df.columns.levels[0])  # 2
    asset_names = raw_df.columns.levels[0].tolist()
    return n_timesteps, n_channels, n_assets, asset_names


def get_dataloaders(raw_df, lookback, gap, horizon, perc_val=0.1, bs=16):
    X, timestamps, y, asset_names, _ = raw_to_Xy(
        raw_df, lookback=lookback, gap=gap, freq="B", horizon=horizon
    )
    dataset = InRAMDataset(X, y, timestamps=timestamps, asset_names=asset_names)
    idxs = [i for i in range(len(dataset))]
    idxs_tr, idxs_val = train_test_split(idxs, test_size=0.15, random_state=42)
    sp_ = len(dataset) - int(len(dataset) * perc_val)
    train_dataloader = RigidDataLoader(dataset, indices=idxs_tr, batch_size=bs)
    val_dataloaders = {"val": RigidDataLoader(dataset, indices=idxs_val, batch_size=bs)}
    return train_dataloader, val_dataloaders, dataset


def build_external_dataloaders(
    external_df,
    assets_df,
    lookback,
    gap,
    horizon,
    perc_val=0.25,
    bs=16,
):
    _, timestamps, y, asset_names, _ = raw_to_Xy(
        assets_df, lookback=lookback, gap=gap, freq="B", horizon=horizon
    )
    X, _, _, _, _ = raw_to_Xy(
        external_df,
        lookback=lookback,
        gap=gap,
        freq="B",
        horizon=horizon,
        receive_raw=True,
    )
    # X_tr, X_val, y_tr, y_val, timestamps_tr, timestamps_val = train_test_split(X, y, timestamps, test_size=0.1, random_state=42)
    dataset = InRAMDataset(X, y, timestamps=timestamps, asset_names=asset_names)
    # tr_dataset = InRAMDataset(X_tr, y_tr, timestamps=timestamps_tr, asset_names=asset_names)
    # val_dataset = InRAMDataset(X_val, y_val, timestamps=timestamps_val, asset_names=asset_names)
    # train_dataloader = DataLoader(tr_dataset, batch_size=bs, num_workers=20)
    # val_dataloader = DataLoader(val_dataset, batch_size=bs, num_workers=20)
    # sp_ = len(dataset) - int(len(dataset)*perc_val)
    # indices_tr = [i for i in range(sp_)]
    # indices_val = [i for i in range(sp_, len(dataset))]
    indices_tr, indices_val = train_test_split(
        range(len(dataset)), test_size=perc_val, random_state=42
    )
    train_dataloader = RigidDataLoader(
        dataset,
        asset_ixs=[i for i in range(X.shape[-1])],
        indices=indices_tr,
        batch_size=bs,
    )
    val_dataloaders = {
        "val": RigidDataLoader(
            dataset,
            asset_ixs=[i for i in range(X.shape[-1])],
            indices=indices_val,
            batch_size=bs,
        )
    }
    return train_dataloader, val_dataloaders, dataset


def get_test_dataloader(external_df, assets_df, lookback, gap, horizon, bs=1):
    _, timestamps, y, asset_names, _ = raw_to_Xy(
        assets_df, lookback=lookback, gap=gap, freq="B", horizon=horizon
    )
    X, _, _, _, _ = raw_to_Xy(
        external_df,
        lookback=lookback,
        gap=gap,
        freq="B",
        horizon=horizon,
        receive_raw=True,
    )
    dataset = InRAMDataset(X, y, timestamps=timestamps, asset_names=asset_names)
    dataloader = RigidDataLoader(
        dataset, asset_ixs=[i for i in range(X.shape[-1])], batch_size=bs
    )
    return dataloader


def get_weights_submission(dataloader, network, device="cuda", dtype=torch.float):
    data = []
    network.eval()
    for X_batch, y_batch, timestamps_batch, asset_names_batch in tqdm(
        dataloader, desc="Getting test predictions..."
    ):
        X_batch, y_batch = X_batch.to(device).to(dtype), y_batch.to(device).to(dtype)
        weights = network(X_batch)
        weights = weights.cpu().detach().numpy()
        weights = general_weights_fixer(weights.squeeze())
        data_batch = [
            {
                "date": timestamps_batch[0],
                **{asset: w_ for asset, w_ in zip(asset_names_batch, weights)},
            }
            # for ts, w in zip(timestamps_batch, weights)
        ]
        data.extend(data_batch)
    df = pd.DataFrame(data)
    df = df.sort_values(by="date")
    df.set_index("date", inplace=True)
    return df


class NumericalRiskNetwork(torch.nn.Module, Benchmark):
    def __init__(
        self,
        n_assets,
        max_weight=1.0,
        force_symmetric=True,
    ):
        super().__init__()
        self.force_symmetric = force_symmetric
        self.matrix = torch.nn.Parameter(torch.eye(n_assets), requires_grad=True)
        self.budgets = torch.nn.Parameter(
            torch.zeros(n_assets), requires_grad=True
        )
        torch.nn.init.uniform_(self.budgets, a=0.0001, b=0.0005)
        # torch.nn.init.uniform_(self.matrix, a=-0.0005, b=0.0005)
        self.portfolio_opt_layer = NumericalRiskBudgeting(n_assets, max_weight)


    def forward(self, x):
        n = len(x)

        covariance = torch.mm(self.matrix, torch.t(self.matrix)) if self.force_symmetric else self.matrix
        budgets = torch.nn.functional.relu(self.budgets, inplace=False)
        budgets = budgets / budgets.sum()

        budgets_all = torch.repeat_interleave(budgets[None, ...], repeats=n, dim=0)
        covariance_all = torch.repeat_interleave(covariance[None, ...], repeats=n, dim=0)
        weights = self.portfolio_opt_layer(covariance_all, budgets_all)
        return weights


class MyNetwork(torch.nn.Module, Benchmark):
    def __init__(
        self,
        n_assets,
        max_weight=0.2,
        force_symmetric=False,
        n_clusters=5,
        n_init=100,
        init="random",
        random_state=None,
    ):
        super().__init__()
        self.force_symmetric = force_symmetric
        self.matrix = torch.nn.Parameter(torch.eye(n_assets), requires_grad=True)
        self.exp_returns = torch.nn.Parameter(
            torch.zeros(n_assets), requires_grad=True
        )  # torch.FloatTensor(n_assets).uniform_(-0.001, 0.001))#torch.tensor(exp_returns), requires_grad=True) #torch.zeros(n_assets), requires_grad=True)
        torch.nn.init.uniform_(self.exp_returns, a=-0.0005, b=0.0005)
        # self.gamma_sqrt = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        # self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        # self.n_clusters = torch.nn.Parameter() # TODO: TRY TO IMPLEMENT THIS DIFFERENTIABLE.
        self.portfolio_opt_layer = NCO(n_clusters, n_init, init, random_state)

    def forward(self, x, external_x=None):
        """
        Receives normal x (returns etc) and external_x,
        which can be composed of as many variables as we want.
        In this case it will be used for sp500, interest rates,
        oil, etc.
        """
        n = len(x)
        covariance = (
            torch.mm(self.matrix, torch.t(self.matrix))
            if self.force_symmetric
            else self.matrix
        )
        exp_returns_all = torch.repeat_interleave(
            self.exp_returns[None, ...], repeats=n, dim=0
        )
        covariance_all = torch.repeat_interleave(
            covariance[None, ...], repeats=n, dim=0
        )
        # gamma_all = torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.gamma_sqrt
        # alpha_all = torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.alpha
        weights = self.portfolio_opt_layer(covariance_all, exp_returns_all)
        return weights


class SoftMaxNetwork(torch.nn.Module, Benchmark):
    def __init__(
        self,
        n_assets,
        max_weight=0.2,
        force_symmetric=True,
    ):
        super().__init__()
        self.force_symmetric = force_symmetric
        self.matrix = torch.nn.Parameter(torch.eye(n_assets), requires_grad=True)
        self.exp_returns = torch.nn.Parameter(torch.zeros(n_assets), requires_grad=True)
        self.temperature = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.gamma_sqrt = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.pre_alloc_layer = NumericalMarkowitz(n_assets, max_weight)
        self.allocation_layer = SoftmaxAllocator(
            n_assets=n_assets,
            temperature=None,
            max_weight=max_weight,
            formulation="variational",
        )

    def forward(self, x):
        n = len(x)
        covariance = (
            torch.mm(self.matrix, torch.t(self.matrix))
            if self.force_symmetric
            else self.matrix
        )
        exp_returns_all = torch.repeat_interleave(
            self.exp_returns[None, ...], repeats=n, dim=0
        )
        covariance_all = torch.repeat_interleave(
            covariance[None, ...], repeats=n, dim=0
        )
        gamma_all = (
            torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.gamma_sqrt
        )
        alpha_all = torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.alpha
        temperatures = torch.ones(n).to(device=x.device, dtype=x.dtype) * self.temperature
        pre_weights = self.pre_alloc_layer(exp_returns_all, covariance_all, gamma_all, alpha_all)
        weights = self.allocation_layer(pre_weights, self.temperature)
        return weights



class ResampleNetwork(torch.nn.Module, Benchmark):
    def __init__(
        self,
        n_assets,
        force_symmetric=True,
        n_clusters=5,
        n_init=10,
        init="random",
        random_state=None,
        max_weight=0.20,
    ):
        super().__init__()
        self.force_symmetric = force_symmetric
        # self.matrix = torch.nn.Parameter(torch.eye(n_assets), requires_grad=True)
        # self.exp_returns = torch.nn.Parameter(torch.zeros(n_assets), requires_grad=True)
        self.gamma_sqrt = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        # self.n_clusters = torch.nn.Parameter() # TODO: TRY TO IMPLEMENT THIS DIFFERENTIABLE.
        self.matrix = torch.nn.Parameter(torch.eye(n_assets), requires_grad=True)
        self.exp_returns = torch.nn.Parameter(torch.zeros(n_assets), requires_grad=True)
        # self.covariance_layer = CovarianceMatrix(sqrt=False, shrinkage_strategy="diagonal")
        # self.collapse_layer = AverageCollapse(collapse_dim=3)
        self.portfolio_opt_layer = Resample(
            allocator=NumericalMarkowitz(
                n_assets, max_weight=max_weight
            ),  # NCO(n_clusters=n_clusters, n_init=n_init, init=init, random_state=random_state),
            n_draws=10,
            n_portfolios=5,
        )

    def forward(self, x):
        # returns = torch.nn.Parameter(x[:, 0 ,:, :], requires_grad=True)
        # exp_returns = x[:, 0, :, :].mean(dim=1)
        # returns = x[:, 0, :, :]
        # exp_returns = returns.mean(dim=1)
        # covmat = self.covariance_layer(returns) #torch.nn.Parameter(self.covariance_layer(returns), requires_grad=True)
        n = len(x)
        covariance = (
            torch.mm(self.matrix, torch.t(self.matrix))
            if self.force_symmetric
            else self.matrix
        )
        exp_returns_all = torch.repeat_interleave(
            self.exp_returns[None, ...], repeats=n, dim=0
        )
        covariance_all = torch.repeat_interleave(
            covariance[None, ...], repeats=n, dim=0
        )
        gamma_all = (
            torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.gamma_sqrt
        )
        alpha_all = torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.alpha
        weights = self.portfolio_opt_layer(
            covariance_all, exp_returns_all, **{"gamma": gamma_all, "alpha": alpha_all}
        )
        return weights

    @property
    def hparams(self):
        """Hyperparameters relevant to construction of the model."""
        return {
            k: v if isinstance(v, (int, float, str)) else str(v)
            for k, v in self._hparams.items()
            if k != "self"
        }


class EconomistNet(torch.nn.Module, Benchmark):
    """
    Network that uses external data to create a richer representation
    of the expected returns, therefore enabling the model to accurately
    change the portfolio weights based on stocks information.
    Later on, more data can be added.

    Parameters
    ----------
    n_external: int
        Number of external variables to use.
    n_assets: int
        Number of assets to trade with.
    n_channels: int
        Number of channels each external variable has.
    hidden_size: int
        Hidden size for LSTM Cells.
    max_weight: float
        Maximum allocation weight for a single asset.
    force_symmetric: bool
        Whether to force the cov matrix to be symmetric.
    p: float
        Percentage of neurons turned 0 due to dropout.
    """

    def __init__(
        self,
        n_external,
        n_assets,
        n_channels=5,
        hidden_size=32,
        max_weight=0.15,
        force_symmetric=True,
        p=0.2,
    ):
        super().__init__()
        self.force_symmetric = force_symmetric
        self.matrix = torch.nn.Parameter(torch.eye(n_assets), requires_grad=True)
        self.exp_returns = torch.nn.Parameter(torch.zeros(n_assets), requires_grad=True)
        self.norm_layer = torch.nn.InstanceNorm2d(n_channels, affine=True)
        self.collapse_external = AverageCollapse()
        self.transform_layer = RNN(n_channels, hidden_size=hidden_size)
        self.dropout_layer = torch.nn.Dropout(p=p)
        self.dropout_layer2 = torch.nn.Dropout(p=p)
        self.time_collapse_layer = AttentionCollapse(n_channels=hidden_size)
        self.conv1 = Conv(
            n_input_channels=hidden_size, n_output_channels=1, method="1D"
        )
        # self.conv2 = Conv(n_input_channels=3, n_output_channels=1, method="1D")
        self.linear_transform = torch.nn.Linear(n_external, n_assets)
        self.linear_2 = torch.nn.Linear(n_external, n_assets)
        self.covariance_layer = CovarianceMatrix(
            sqrt=False, shrinkage_strategy="diagonal"
        )
        self.gamma_sqrt = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.preliminar_weights_layer = NumericalMarkowitz(
            n_assets, max_weight=max_weight
        )

    def forward(self, x):
        """
        Performs a forward pass. For that, we first create covariance matrix.
        Then, we perform all transformations to external variables and finally
        we multiply the weights received from NumericalMarkowitz (as in ThorpeNet)
        and then we change this result for this batch taking into account the
        external variables.

        Parameters
        ----------
        x: torch.tensor
            Tensor of shape (n_samples, n_channels, loockback, n_assets)

        Returns
        -------
        weights: torch.tensor
            Tensor of shape (n_samples, n_assets) representing the optimal portfolio weights.
        """
        n = len(x)
        # x = self.collapse_external(x)
        x = self.norm_layer(x)

        # Covmat
        rets = x[:, 3, :, :]
        rets = self.linear_2(rets)
        cov = self.covariance_layer(rets)
        x = self.transform_layer(x)
        x = self.dropout_layer(x)
        x = self.time_collapse_layer(x)
        x = self.conv1(x)
        x = self.dropout_layer2(x)
        # x = self.conv2(x)
        x = self.linear_transform(x)
        # covariance = torch.mm(self.matrix, torch.t(self.matrix)) if self.force_symmetric else self.matrix
        # covariance_all = torch.repeat_interleave(covariance[None, ...], repeats=n, dim=0)
        gamma_all = (
            torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.gamma_sqrt
        )
        alpha_all = torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.alpha
        weights = self.preliminar_weights_layer(
            x.view(x.shape[0], x.shape[-1]), cov, gamma_all, alpha_all
        )
        return weights

    # @property
    # def hparams(self):
    #    """Hyperparameters relevant to construction of the model."""
    #    return {k: v if isinstance(v, (int, float, str)) else str(v) for k, v in self._hparams.items() if k != 'self'}


def train_model(
    network, train_dataloader, val_dataloaders, optimizer, callbacks, epochs=20, device="cpu", loss_="sharpe"
):
    if loss_ == "sharpe":
        loss = SharpeRatio(returns_channel=0)
    else:
        loss = MaximumDrawdown(returns_channel=0)

    benchmarks = {"1overN": OneOverN()}
    metrics = {"drawdown": MaximumDrawdown(returns_channel=0), "sharpe": SharpeRatio(returns_channel=0)}
    run = Run(
        network,
        loss,
        train_dataloader,
        val_dataloaders=val_dataloaders,
        metrics=metrics,
        # benchmarks=benchmarks,
        device=torch.device(device),
        optimizer=optimizer,
        callbacks=callbacks,
    )
    history = run.launch(n_epochs=epochs)
    return run


def get_predictions(run, n_channels, lookback, n_assets):
    w_pred = (
        run.models["main"]
        .to("cpu")(torch.ones(1, n_channels, lookback, n_assets).cpu())
        .cpu()
        .detach()
        .numpy()
        .squeeze()
    )
    return w_pred
