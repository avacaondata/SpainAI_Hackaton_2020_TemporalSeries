# SPAINAI HACKATON 2021: TEMPORAL SERIES CHALLENGE

In this repository you will find the main code used for obtaining the first prize in the Time Series Competition from the SpainAI Hackaton 2020. The code scripts are not fully cleaned, as I haven't find the time for cleaning and documenting them correctly.

## WHAT WAS THE CHALLENGE ABOUT

![Alt text](imgs/diagram_series_temporales.png?raw=true "Diagram")

For the challenge we had a total of 96 assets, for which we had their temporal series, containing their OHLCV hourly values for a year and a half, approximately, from December 2018 to June 2020. The objective of the challenge was to create a ML system capable of adjusting the weights of the wallet such that we maximize the Sharpe Ratio obtained from August 2020 to December 2020. There were several issues with this data:

* Lots of missing values in train data. 
* Almost 2 month gap between train and test data.
* Little movement from hour to hour: stable assets (good for wallet value stability, bad for arbitrage).
* Out strategy cannot depend on last prices, therefore it cannot be dynamic, which is very distant from the strategy we would follow in a real world setting, where we'd have the last prices for making the current decision.
