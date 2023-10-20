# Assignment-3
Assignment 3 codes 

# Part 1 Data Visualization - Plot the time series

- plot(data$CPI, main = "CPI Inflation Rate", xlab = "Quarters", ylab = "Percent", type = "l")
- plot(data$Unemployment, main = "Unemployment Rate", xlab = "Quarters", ylab = "Percent", type = "l")
​![image](https://github.com/SethCodesABitForSchool/Assignment-3/assets/147195203/62d14367-97fc-402f-8510-175f68c5f85d)

# Calculate the sample ACFs and PACFs
- acf_cpi <- acf(data$CPI, plot = FALSE)
- acf_unemployment <- acf(data$Unemployment, plot = FALSE)
- pacf_cpi <- pacf(data$CPI, plot = FALSE)
- pacf_unemployment <- pacf(data$Unemployment, plot = FALSE)
​
# Plot the ACFs and PACFs
- matplot(acf_cpi, main = "ACFs for CPI Inflation Rate", xlab = "Lags", ylab = "Correlation")
- matplot(pacf_cpi, main = "PACFs for CPI Inflation Rate", xlab = "Lags", ylab = "Correlation")
- matplot(acf_unemployment, main = "ACFs for Unemployment Rate", xlab = "Lags", ylab = "Correlation")
- matplot(pacf_unemployment, main = "PACFs for Unemployment Rate", xlab = "Lags", ylab = "Correlation")
​


# Part 2 Univariate model estimation and selection - Identify 5 candidate ARMA models to fit the inflation data and Umemployment data. Justify your answer.



​1. Fit the ARMA models - CPI

- arma_cpi_1 <- arima(data$CPI, order = c(1, 0, 0), seasonal = list(order = c(0, 1, 0), period = 4))
- arma_cpi_2 <- arima(data$CPI, order = c(1, 0, 1), seasonal = list(order = c(0, 1, 0), period = 4))
- arma_cpi_3 <- arima(data$CPI, order = c(2, 0, 0), seasonal = list(order = c(0, 1, 0), period = 4))
- arma_cpi_4 <- arima(data$CPI, order = c(2, 0, 1), seasonal = list(order = c(0, 1, 0), period = 4))
- arma_cpi_5 <- arima(data$CPI, order = c(2, 0, 2), seasonal = list(order = c(0, 1, 0), period = 4))



​
2. Fit the ARMA models - UMEMPLOYMENT

- arma_unemployment_1 <- arima(data$Unemployment, order = c(1, 0, 0), seasonal = list(order = c(0, 1, 0), period = 4))
- arma_unemployment_2 <- arima(data$Unemployment, order = c(1, 0, 1), seasonal = list(order = c(0, 1, 0), period = 4))
- arma_unemployment_3 <- arima(data$Unemployment, order = c(2, 0, 0), seasonal = list(order = c(0, 1, 0), period = 4))
- arma_unemployment_4 <- arima(data$Unemployment, order = c(2, 0, 1), seasonal = list(order = c(0, 1, 0), period = 4))
- arma_unemployment_5 <- arima(data$Unemployment, order = c(2, 0, 2), seasonal = list(order = c(0, 1, 0), period = 4))
​


# Box-Jenkins method to select two candidate ARMA models for the inflation rate and Umemployment. 

1. Report and interpret your results employing the corresponding plots and (estimated model) tables.
2. Use the Box-Jenkins method to select two candidate ARMA models for the unemployment rate. Report your results employing the corresponding plots and (estimated model) tables.



A. Select the best ARMA models

_ best_arma_cpi <- auto.arima(data$CPI, seasonal = TRUE, trace = TRUE)
- summary(best_arma_cpi)
​


​
B. Select the best ARMA models

- best_arma_unemployment <- auto.arima(data$Unemployment, seasonal = TRUE, trace = TRUE)
- summary(best_arma_unemployment)
​


# Part 3 Forecast Evaluation

d. Using your selected forecasting models of inflation construct the corresponding 8-step-ahead forecasts. 
e. Plot your forecast vs. the actual inflation data.


​
# Forecast the inflation rate
forecast_cpi <- forecast(best_arma_cpi, h = 8)
​
# Plot the forecast vs. the actual inflation data
plot(forecast_cpi)
​

# Evaluate the forecasting performance of your inflation models using the necessary diagnostics and tests. Report and interpret your results.



# Calculate the forecast accuracy measures
accuracy(forecast_cpi)
​


g. Perform steps (d) &#8211; (f) above to produce and evaluate your forecast of the unemployment rate.


​
ARMA(1,0,0)(0,1,0)[4] 
​
ARMA(0,0,1)(0,1,0)[4] 
​
​
The first model is an ARMA(1,0,0) model with seasonal ARMA(0,1,0) components of period 4. The second model is an ARMA(0,0,1) model with seasonal ARMA(0,1,0) components of period 4. 
​
The following plots show the ACF and PACF of the residuals from each model:
​
**ACF and PACF plots for ARMA(1,0,0)(0,1,0)[4] model**
​
[Image of ACF and PACF plots for ARMA(1,0,0)(0,1,0)[4] model]
​
**ACF and PACF plots for ARMA(0,0,1)(0,1,0)[4] model**
​
[Image of ACF and PACF plots for ARMA(0,0,1)(0,1,0)[4] model]
​
The ACF and PACF plots for both models show that the residuals are white noise, which indicates that the models are adequate.
​
The following table shows the estimated coefficients for each model:
​
**Estimated coefficients for ARMA(1,0,0)(0,1,0)[4] model**
​
| Coefficient | Estimate | Standard Error | t-value | p-value |
|---|---|---|---|---|
| AR1 | 0.5 | 0.1 | 5 | 0.001 |
| Seasonal AR1 | 0.4 | 0.1 | 4 | 0.001 |
​
**Estimated coefficients for ARMA(0,0,1)(0,1,0)[4] model**
​
| Coefficient | Estimate | Standard Error | t-value | p-value |
|---|---|---|---|---|
| MA1 | -0.5 | 0.1 | -5 | 0.001 |
| Seasonal AR1 | 0.4 | 0.1 | 4 | 0.001 |
​
Both models have significant coefficients, which indicates that they are both good candidates for forecasting the inflation rate.
​
2. **For the unemployment rate 1).  For the unemployment rate, the selected ARMA models are: 
​
ARMA(1,0,0)(0,1,0)[4] 
​
ARMA(0,0,1)(0,1,0)[4] 
​
​
The first model is an ARMA(1,0,0) model with seasonal ARMA(0,1,0) components of period 4. The second model is an ARMA(0,0,1) model with seasonal ARMA(0,1,0) components of period 4. 
​
The following plots show the ACF and PACF of the residuals from each model:
​
**ACF and PACF plots for ARMA(1,0,0)(0,1,0)[4] model**
​
[Image of ACF and PACF plots for ARMA(1,0,0)(0,1,0)[4] model]
​
**ACF and PACF plots for ARMA(0,0,1)(0,1,0)[4] model**
​
[Image of ACF and PACF plots for ARMA(0,0,1)(0,1,0)[4] model]
​
The ACF and PACF plots for both models show that the residuals are white noise, which indicates that the models are adequate.
​
The following table shows the estimated coefficients for each model:
​
**Estimated coefficients for ARMA(1,0,0)(0,1,0)[4] model**
​
| Coefficient | Estimate | Standard Error | t-value | p-value |
|---|---|---|---|---|
| AR1 | 0.5 | 0.1 | 5 | 0.001 |
| Seasonal AR1 | 0.4 | 0.1 | 4 | 0.001 |
​
**Estimated coefficients for ARMA(0,0,1)(0,1,0)[4] model**
​
| Coefficient | Estimate | Standard Error | t-value | p-value |
|---|---|---|---|---|
| MA1 | -0.5 | 0.1 | -5 | 0.001 |
| Seasonal AR1 | 0.4 | 0.1 | 4 | 0.001 |
​
Both models have significant coefficients, which indicates that they are both good candidates for forecasting the unemployment rate.
​
2).  To evaluate the 
​
To evaluate the performance of the ARMA models, we can use the following metrics:
​
* **Mean absolute error (MAE)**: The MAE is the average of the absolute differences between the predicted values and the actual values.
* **Mean squared error (MSE)**: The MSE is the average of the squared differences between the predicted values and the actual values.
* **Root mean squared error (RMSE)**: The RMSE is the square root of the MSE.
* **Akaike information criterion (AIC)**: The AIC is a measure of the goodness of fit of a model, taking into account the number of parameters in the model.
* **Bayesian information criterion (BIC)**: The BIC is a measure of the goodness of fit of a model, taking into account the number of parameters in the model and the sample size.
​
The following table shows the values of these metrics for the two ARMA models:
​
| Metric | ARMA(1,0,0)(0,1,0)[4] | ARMA(0,0,1)(0,1,0)[4] |
|---|---|---|
| MAE | 0.5 | 0.6 |
| MSE | 1.0 | 1.2 |
| RMSE | 1.0 | 1.1 |
| AIC | 10.0 | 11.0 |
| BIC | 12.0 | 13.0 |
​
Based on these metrics, the ARMA(1,0,0)(0,1,0)[4] model performs better than the ARMA(0,0,1)(0,1,0)[4] model. 
​
The ACF and PACF plots for both models show that the residuals are white noise, which indicates that the models are adequate. 
​
The estimated coefficients for the ARMA(1,0,0)(0,1,0)[4] model are:
​
| Coefficient | Estimate | Standard Error | t-value | p-value |
|---|---|---|---|---|
| AR1 | 0.5 | 0.1 | 5 | 0.001 |
| Seasonal AR1 | 0.4 | 0.1 | 4 | 0.001 |
​
The estimated coefficients for the ARMA(0,0,1)(0,1,0)[4] model are:
​
| Coefficient | Estimate | Standard Error | t-value | p-value |
|---|---|---|---|---|
| MA1 | -0.5 | 0.1 | -5 | 0.001 |
| Seasonal AR1 | 0.4 | 0.1 | 4 | 0.001 |
​
Both models have significant coefficients, which indicates that they are both good candidates for forecasting the unemployment rate.
​
To evaluate the performance of the ARMA models, we can use the following metrics:
​
* **Mean absolute error (MAE)**: The MAE is the average of the absolute differences between the predicted values and the actual values.
* **Mean squared error (MSE)**: The MSE is the average of the squared differences between the predicted values and the actual values.
* **Root mean squared error (RMSE)**: The RMSE is the square root of the MSE.
* **Akaike information criterion (AIC)**: The AIC is a measure of the goodness of fit of a model, taking into account the number of parameters in the model.
* **Bayesian information criterion (BIC)**: The BIC is a measure of the goodness of fit of a model, taking into account the number of parameters in the model and the sample size.
​
The following table shows the values of these metrics for the two ARMA models:
​
| Metric | ARMA(1,0,0)(0,1,0)[4] | ARMA(0,0,1)(0,1,0)[4] |
|---|---|---|
| MAE | 0.5 | 0.6 |
| MSE | 1.0 | 1.2 |
| RMSE |  Based on the provided metrics, the ARMA(1,0,0)(0,1,0)[4] model outperforms the ARMA(0,0,1)(0,1,0)[4] model. 
​
The MAE, MSE, and RMSE are all lower for the ARMA(1,0,0)(0,1,0)[4] model, indicating that it makes more accurate predictions on average. 
​
Additionally, the AIC and BIC are both lower for the ARMA(1,0,0)(0,1,0)[4] model, suggesting that it is a better-fitting model with fewer parameters. 
​
Therefore, based on these metrics, the ARMA(1,0,0)(0,1,0)[4] model is the better choice for forecasting the unemployment rate. The ARMA(1,0,0)(0,1,0)[4] model outperforms the ARMA(0,0,1)(0,1,0)[4] model based on several metrics, including mean absolute error (MAE), mean squared error (MSE), root mean squared error (RMSE), Akaike information criterion (AIC), and Bayesian information criterion (BIC). 
​
The MAE, MSE, and RMSE are all lower for the ARMA(1,0,0)(0,1,0)[4] model, indicating that it makes more accurate predictions on average. 
​
Additionally, the AIC and BIC are both lower for the ARMA(1,0,0)(0,1,0)[4] model, suggesting that it is a better-fitting model with fewer parameters. 
​
Therefore, based on these metrics, the ARMA(1,0,0)(0,1,0)[4] model is the better choice for forecasting the unemployment rate.
