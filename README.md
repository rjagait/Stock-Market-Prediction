# Stock-Market-Prediction

This project deals with predicting the opening BID, ASK and PRICE using daily Apple stock data of 4 years. This data would be useful for traders to determine if a stock should be traded as a future, forward, option or some other derivative as they would be able to see the future trend.

## Data Pre-processing and Feature Engineering
1. Check for missing values - no missing values
2. Check if categorical data, encoding needed - no categorical data
3. Check Feature Importance - All features had high significance, so not dropping
4. Split data into train, validation and test in the ratio 60:20:20
5. Feature Scaling - We are normalizing data between zero and one.

## Prediction Models
1. Autoregressive Integrated Moving Average models (ARIMA)
2. Long Short-Term Memory (LSTM)
3. Gated Recurrent Unit (GRU)

## Model Optimization
Bayesian Optimization of GRU and LSTM models.

## Accuracy Measures for Comparison
1. Mean Absolute Error (MAE)
2. Root Mean Squared Error (RMSE)
3. Mean Forecast Error (MFE)

![accuracy comparison3](https://user-images.githubusercontent.com/55213734/81438718-336b7600-913b-11ea-8463-b1b5afcd06ea.PNG)

Even the less complex models like ARIMA can show better performance than the complex neural netwrok models on some datasets.
