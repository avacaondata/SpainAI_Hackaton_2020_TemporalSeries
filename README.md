# SPAINAI HACKATON 2021: TEMPORAL SERIES CHALLENGE

In this repository you will find the main code used for obtaining the first prize in the Time Series Competition from the SpainAI Hackaton 2020. The code scripts are not fully cleaned, as I haven't find the time for cleaning and documenting them correctly.

## WHAT WAS THE CHALLENGE ABOUT

![Alt text](imgs/diagram_series_temporales.png?raw=true "Diagram")

For the challenge we had a total of 96 assets, for which we had their temporal series, containing their OHLCV hourly values for a year and a half, approximately, from December 2018 to June 2020. The objective of the challenge was to create a ML system capable of adjusting the weights of the wallet such that we maximize the Sharpe Ratio obtained from August 2020 to December 2020. There were several issues with this data:

* Lots of missing values in train data. 
* Almost 2 month gap between train and test data.
* Little movement from hour to hour: stable assets (good for wallet value stability, bad for arbitrage).
* Out strategy cannot depend on last prices, therefore it cannot be dynamic, which is very distant from the strategy we would follow in a real world setting, where we'd have the last prices for making the current decision.

## GENERAL CODE SUMMARY

The script [data_utils.py](data_utils.py) has different functions for processing time series data, in particular assets data. In [download_data.py](download_data.py) and [download_data_yahoo.py](download_data_yahoo.py) we can find code used for downloading data from investpy and yahoo, respectively. With [more_stocks.py](more_stocks.py), we can fill the complementary data folder with data from cryptocurrencies, oil, sp500 index, gold, silver, forex (different pairs) and more. [submission_utils.py](submission_utils.py) and [general_utils.py](general_utils.py) are useful for fixing the weights received from the different methods and have other utilities for making the submissions as required by the competition terms. The rest of the code will be mentioned later or is not important enough to mention it. 

## SOLUTION

* **Benchmark: Markowitz' Portfolio Theory**: The Markowitz' Modern Portfolio Theory is described in [this link](https://www.investopedia.com/terms/m/modernportfoliotheory.asp). Basically, it tries to solve the optimization problem of finding the optimal portfolio weights to maximize Sharpe ratio over a period, using historical mean returns and historical covariance matrix. For this method we need to do the inverse of the covariance matrix, which is a very unstable operation. This is one of its drawbacks. The sharpe ratio obtained with this method was (from now on, the **score**): **1.97**. The code for this is in [portfolio_optimization.py](portfolio_optimization.py).

![Alt text](imgs/markowitz_curve.png?raw=true "Markowitz")

* **Reinforcement Learning**: We use external data as the signal, as we won't have previous prices at test time. Darwins prices data is only used for computing the reward. There's a significant scarcity of free and available minute or hourly data for indexes or commodities, which makes this very hard to do. We have to work with few external assets: we barely have signal. However, if we use daily data, with more availability and more signal, we wouldn't have enough data for training a RL algorithm. The environment for training this RL agent is in [env.py](env.py), and the script used for training a [PPO](https://openai.com/blog/openai-baselines-ppo/) is in [train_rl.py](train_rl.py). **Score: 0.30**

![Alt text](imgs/RL.png?raw=true "Markowitz")

* **MLFINLAB + PORTFOLIOLAB**: With these 2 libraries, which can be found [here](https://mlfinlab.readthedocs.io/en/latest/index.html) and [here](https://mlfinlab.readthedocs.io/en/latest/getting_started/portfoliolab.html), we have many different algorithms for optimizing portfolio, as well as different methods for estimating returns and covariance matrix. I tried different algorithms from these libraries, with the following results:
    1. Nested Cluster Optimization (code in [notebooks/NCO.ipynb](notebooks/NCO.ipynb) and [notebooks/HRP.ipynb](notebooks/HRP.ipynb)). **Score: 3.75**
    2. Robust Bayesian Allocation (code in [notebooks/try_bayesian_alloc.ipynb](notebooks/try_bayesian_alloc.ipynb)). **Score: 1.06**
    3. Hierarchical Risk Parity (code in [notebooks/HRP.ipynb](notebooks/HRP.ipynb)). **Score: 4.52**
From these methods I learned that distributed risk management (not dependand on overusing a single stable asset) raises Sharpe Ratio a lot.

![Alt text](imgs/hrp.png?raw=true "HRP")


* **DeepDow: Pytorch Deep Learning + Convex Optimization**: [This library](https://deepdow.readthedocs.io/en/latest/index.html) uses cvxpy for the optimization layer, and has multiple intermediary layers between the input and the optimization layer, using Pytorch. The advantage of using PyTorch before the numerical optimization layer is that we can learn in batches, we can be creative and create multiple ways of performing feature engineering, and we can use multiple loss functions and see which ones work best for our task at hand. I used the following approaches:
    1. Modifying the library source code (my version is in [deepdow/](deepdow/)) for enabling the use of external variables as predictors. Following this, I designed the *EconomistNet*, which can be found in [opt_nets.py](opt_nets.py). Its architecture is based on InceptionTime, using convolutional layers to extract useful information from time series, as well as a recurrent part. The external daily data used for this was: all Sp500 (all companies forming it), Cryptocurrencies, Forex, Gold, other commodities, etc. **Score: 3.76**.
    2. ThorpeNet: This is a special type of Network in which we learn &alpha;, &gamma; , &mu; and &Sigma; with backpropagation and then use Numerical Markowitz as the optimization layer. Using different loss functions and configurations we can get different results:
        - With all historic data, using SharpeRatio loss, **Score: 4.50**
        - With all historic data, using MaximumDrawdown loss, **Score: 4.75**
        - With data from pandemic, using MaximumDrawdown loss, **Score: 4.97**
    3. At this moment, I discovered that Darwinex had its data public, therefore test data was also available. The organization authorizes the use of this data. "Interestingly", when I started using it, I replicated, one by one, the results obtained by the competitors I had just above, depending on the configuration used: 8, 6.67, 5.61, respectively. With pandemic data, maximizing SharpeRatio, **Score: 8.38**.

## METHODS COMPARISON

![Alt text](imgs/results_seriestemporales.png?raw=true "Markowitz")
