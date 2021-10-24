# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 12:10:21 2021

@author: Nishan Kapoor
"""
#########      question 1      ###########

import pandas as pd
airlines = pd.read_excel(r"E:\Forecasting - Time series\Airlines Data.xlsx")

# Pre processing
import numpy as np

#from datetime import date


airlines['Month'] = pd.to_datetime(airlines['Month'], format='%m').dt.month_name().str.slice(stop=3)

airlines["t"] = np.arange(1,97)

airlines["t_square"] = airlines["t"] * airlines["t"]
airlines["log_Passengers"] = np.log(airlines["Passengers"])
airlines.columns


p = airlines["Month"][0]
p[0:3]

airlines['months']= 0

for i in range(96):
    p = airlines["Month"][i]
    airlines['months'][i]= p[0:3]
    
month_dummies = pd.DataFrame(pd.get_dummies(airlines['months']))
airlines1 = pd.concat([airlines, month_dummies], axis = 1)

# Visualization - Time plot
airlines1.Passengers.plot()

# Data Partition
Train = airlines1.head(84)
Test = airlines1.tail(12)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers ~ t', data = Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers']) - np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_Passengers ~ t', data = Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers']) - np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Passengers ~ t + t_square', data = Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t", "t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers']) - np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('Passengers ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers']) - np.array(pred_add_sea))**2))
rmse_add_sea

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Passengers ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers']) - np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

################## Additive Seasonality Quadratic Trend ############################

add_sea_Quad = smf.ols('Passengers ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality Linear Trend  ###########

Mul_Add_sea = smf.ols('log_Passengers ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse = pd.DataFrame(data)
table_rmse

# 'rmse_add_sea' has the least value among the models prepared so far Predicting new values 
predict_data = pd.read_excel(r"E:\Forecasting - Time series\new_airlines.xlsx")

predict_data['Month'] = pd.to_datetime(predict_data['Month'], format='%m').dt.month_name().str.slice(stop=3)

model_full = smf.ols('Passengers ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data=airlines1).fit()

pred_new  = pd.Series(model_full.predict(predict_data))
pred_new

predict_data["forecasted_Passengers"] = pd.Series(pred_new)


# Autoregression Model (AR)
# Calculating Residuals from best model applied on full data
# AV - FV
full_res = airlines1.Passengers - model_full.predict(airlines1)

# ACF plot on residuals
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(full_res, lags = 12)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.

# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series 
tsa_plots.plot_pacf(full_res, lags=12)

# Alternative approach for ACF plot
# from pandas.plotting import autocorrelation_plot
# autocorrelation_ppyplot.show()
                          

# AR model
from statsmodels.tsa.ar_model import AutoReg
model_ar = AutoReg(full_res, lags=[1])
# model_ar = AutoReg(Train_res, lags=12)
model_fit = model_ar.fit()

print('Coefficients: %s' % model_fit.params)


pred_res = model_fit.predict(start=len(full_res), end=len(full_res)+len(predict_data)-1, dynamic=False)
pred_res.reset_index(drop=True, inplace=True)


# The Final Predictions using ASQT and AR(1) Model
final_pred = pred_new + pred_res
final_pred


#############################################################################

############   question 1 - ARIMA Model      ##############

import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA


tsa_plots.plot_acf(airlines.Passengers, lags = 12)

# ARIMA with MA = 6
model1 = ARIMA(airlines.Passengers, order = (1,1,6))
res1 = model1.fit()
print(res1.summary())

model2 = ARIMA(airlines.Passengers, order = (1,1,5))


p=1
q=0
d=1

pdq=[]
for q in range(7):
    try:
        model = ARIMA(airlines.Passengers, order = (p, d, q))
        x1= p,d,q
        pdq.append(x1)
    except:
        pass
            
keys = pdq


# one-step out of sample forecast
start_index = len(airlines)
end_index = len(airlines)

# Forecast next 1 month value
forecast = res1.predict(start=start_index, end=end_index)

print('Forecast: %f' % forecast)


# Forecast for next 60 months or 5 years
start_index = len(airlines) + 1
end_index = start_index + 60
forecast = res1.predict(start=start_index, end=end_index)

print(forecast)


#############################################################################################

########################################################################################

#######     question 3     #######

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # Holt Winter's Exponential Smoothing


plastic = pd.read_csv(r"E:\Forecasting - Time series\PlasticSales.csv")

plastic.Sales.plot() # time series plot 

# Splitting the data into Train and Test data
# Recent 4 time period values are Test data
Train = plastic.head(48)
Test = plastic.tail(12)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# Moving Average for the time series
mv_pred = plastic["Sales"].rolling(12).mean()
mv_pred.tail(12)
MAPE(mv_pred.tail(12), Test.Sales)


# Plot with Moving Averages
plastic.Sales.plot(label = "org")
for i in range(2, 9, 2):
    plastic["Sales"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)


# Time series decomposition is the process of separating data into its core components.
# Time series decomposition plot using Moving Average
decompose_ts_add = seasonal_decompose(plastic.Sales, model = "additive", period = 12)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

decompose_ts_mul = seasonal_decompose(plastic.Sales, model = "multiplicative", period = 12)
decompose_ts_mul.plot()

# ACF and PACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(plastic.Sales, lags = 4)
tsa_plots.plot_pacf(plastic.Sales, lags=4)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.

# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_ses, Test.Sales) 

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hw, Test.Sales) 

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"], seasonal = "add", trend = "add", seasonal_periods = 12).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_add_add, Test.Sales) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"], seasonal = "mul", trend = "add", seasonal_periods = 12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_mul_add, Test.Sales) 


# Final Model on 100% Data
hwe_model_add_add = ExponentialSmoothing(plastic["Sales"], seasonal = "add", trend = "add", seasonal_periods = 12).fit()

# Load the new data which includes the entry for future 4 values
new_data = pd.read_csv(r"E:\Forecasting - Time series\updated_plasticSales1.csv")

newdata_pred = hwe_model_add_add.predict(start = new_data.index[0], end = new_data.index[-1])
newdata_pred

#####################################################################################

########       question 2   ########

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # Holt Winter's Exponential Smoothing


cocacola = pd.read_excel("E:\Forecasting - Time series\CocaCola_Sales_Rawdata.xlsx")

cocacola.Sales.plot() # time series plot 

# Splitting the data into Train and Test data
# Recent 4 time period values are Test data
Train = cocacola.head(38)
Test = cocacola.tail(4)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# Moving Average for the time series
mv_pred = cocacola["Sales"].rolling(4).mean()
mv_pred.tail(4)
MAPE(mv_pred.tail(4), Test.Sales)


# Plot with Moving Averages
cocacola.Sales.plot(label = "org")
for i in range(2, 9, 2):
    cocacola["Sales"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)


# Time series decomposition is the process of separating data into its core components.
# Time series decomposition plot using Moving Average
decompose_ts_add = seasonal_decompose(cocacola.Sales, model = "additive", period = 4)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

decompose_ts_mul = seasonal_decompose(cocacola.Sales, model = "multiplicative", period = 4)
decompose_ts_mul.plot()

# ACF and PACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(cocacola.Sales, lags = 4)
tsa_plots.plot_pacf(cocacola.Sales, lags=4)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.

# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_ses, Test.Sales) 

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hw, Test.Sales) 

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_add_add, Test.Sales) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_mul_add, Test.Sales) 


# Final Model on 100% Data
hwe_model_add_add = ExponentialSmoothing(cocacola["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()

# Load the new data which includes the entry for future 4 values

new_data = pd.read_excel("E:\Forecasting - Time series\Cocacola_new_predicted.xlsx")

newdata_pred = hwe_model_add_add.predict(start = new_data.index[0], end = new_data.index[-1])
newdata_pred

#############################################################################################

########      question 4      #######

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # Holt Winter's Exponential Smoothing


solar = pd.read_csv("E:\Forecasting - Time series\solarpower_cumuldaybyday2.csv")

solar.cum_power.plot() # time series plot 

# Splitting the data into Train and Test data
# Recent 4 time period values are Test data
Train = solar.head(2528)
Test = solar.tail(30)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# Moving Average for the time series
mv_pred = solar["cum_power"].rolling(30).mean()
mv_pred.tail(30)
MAPE(mv_pred.tail(30), Test.cum_power)


# Plot with Moving Averages
solar.cum_power.plot(label = "org")
for i in range(100, 1000, 200):
    solar["cum_power"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)


# Time series decomposition is the process of separating data into its core components.
# Time series decomposition plot using Moving Average
decompose_ts_add = seasonal_decompose(solar.cum_power, model = "additive", period = 30)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

decompose_ts_mul = seasonal_decompose(solar.cum_power, model = "multiplicative", period = 30)
decompose_ts_mul.plot()

# ACF and PACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(solar.cum_power, lags = 10)
tsa_plots.plot_pacf(solar.cum_power, lags=10)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.

# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["cum_power"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_ses, Test.cum_power) 

# Holt method 
hw_model = Holt(Train["cum_power"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hw, Test.cum_power) 

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["cum_power"], seasonal = "add", trend = "add", seasonal_periods = 30).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_add_add, Test.cum_power) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["cum_power"], seasonal = "mul", trend = "add", seasonal_periods = 30).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_mul_add, Test.cum_power) 


# Final Model on 100% Data
hwe_model_add_add = ExponentialSmoothing(solar["cum_power"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()

# Load the new data which includes the entry for future 4 values

new_data = pd.read_csv(r"E:\Forecasting - Time series\new_solarpower.csv")

newdata_pred = hwe_model_add_add.predict(start = new_data.index[0], end = new_data.index[-1])
newdata_pred


##################     completed       ####################################

