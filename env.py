import datetime
import random

import gym
import numpy as np
import pandas as pd
from empyrical import sharpe_ratio
from gym import spaces
from tqdm import tqdm

from submission_utils import general_weights_fixer


class MyEnv(gym.Env):
    """
    Environment for a trading agent that only chooses
    what amount of the portfolio to allocate in each
    asset.
    """

    def __init__(self, obs_file, assets_file, n_assets=96):
        super(MyEnv, self).__init__()
        self.obs_df = pd.read_csv(obs_file, parse_dates=["date"], index_col="date")
        self.assets_df = pd.read_csv(
            assets_file, parse_dates=["date"], index_col="date"
        )
        # self.assets_df.fillna(0, inplace=True)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_assets,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(110,))

    def reset(
        self,
    ):
        self.game_df, self.assets_game = (
            self.obs_df.copy(),
            self.assets_df.copy(),
        )  # self._get_game_df()
        self.current_step = 0
        self.steps_left = self.game_df.shape[0] - 1
        self.current_positions = [1 / self.assets_df.shape[1]] * self.assets_df.shape[
            1
        ]  # [0.]*self.assets_df.shape[1]
        self.wallet_values = [self._wallet_value()]
        return self._next_obs()

    def step(self, action):
        action = np.round(action, decimals=3)
        action = np.clip(action, 0, 1)
        action = action / sum(action)
        action = general_weights_fixer(action)
        penalize_reward = False
        if all([act == 0 for act in action]):
            penalize_reward = True
            reward = -1000
        self.current_positions = action
        self.wallet_values.append(self._wallet_value())
        self.steps_left -= 1
        self.current_step += 1
        done = False
        if self.steps_left == 0:
            done = True
            self.reset()
        obs = self._next_obs()
        if not penalize_reward:
            reward = self._compute_reward()
            reward = reward if not np.isnan(reward) else -1000  # 0.0
        return (obs, reward, done, {})

    def _compute_reward(
        self,
    ):
        returns = np.diff(np.log(self.wallet_values))
        sharpe = sharpe_ratio(returns, annualization=365 * 24)
        return sharpe

    def _get_game_df(
        self,
    ):
        last_date = self.assets_df.index[-1] - datetime.timedelta(days=28 * 6)
        fechas_previas = self.assets_df.loc[self.assets_df.index <= last_date].index
        start_date = random.choice(fechas_previas)
        obsdf = self.obs_df.loc[self.obs_df.index >= start_date]
        assetsdf = self.assets_df.loc[self.assets_df.index >= start_date]
        return obsdf, assetsdf

    def _next_obs(
        self,
    ):
        return self.game_df.iloc[self.current_step, :].values

    def _wallet_value(
        self,
    ):
        return np.dot(
            self.current_positions, self.assets_game.iloc[self.current_step, :].values
        )
