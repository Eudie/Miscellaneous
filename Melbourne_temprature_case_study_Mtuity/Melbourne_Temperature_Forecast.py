
# coding: utf-8

# # This notebook is to forecast temperature for future dates in Melbourne using historical data
# 
# ## The structure of case study is as follows:
# 
# 
# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. Loading and preprocessing data
# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. Exploratory Analysis
# #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;i. Univariate distribution
# #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ii. Bivariate analysis
# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. Missing value imputation
# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. Building time series model (Seasonal ARIMA)
# #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;i. Finding Optimum hyper-parameters using grid search
# #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ii. Building Model
# #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;iii. Checking for the pattern in residuals
# #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;iv. Validation
# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5. Forecasting for the future dates
# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6. Next steps for improving the results
# 

# In[1]:

import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[2]:

warnings.filterwarnings("ignore")
data = pd.read_csv("melbourne_weather.csv", header = 1)
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data.sort_index(inplace=True)
data.head()


# # Exploratory Analysis and Preprocessing
# 
# ## Let's try to understand data
# 
# In this step first, we will look for the values in all columns. After that, we will look for the distribution. Then we will treat any missing values and outlier
# 

# In[3]:

data.index


# In[4]:

data.describe()


# In[5]:

print(data['conditions'].value_counts())
print('\n')
print(data['events'].value_counts())
print('\n')
print(data['precip'].value_counts())
print('\n')
print(data['winddir'].value_counts())


# ### Based on our observation we can remove the following column
# 
# 0        : Contains just index numbers
# 
# windchill: All Nan
# 
# gustspeed: All Nan
# 
# heatindex: All Nan
# 
# events   : Only one observation rest all Nan
# 
# precip   : All Nan (in string format "-")
# 
# time     : Already captured in datetime
# 
# date     : Already captured in datetime

# In[6]:

data.drop(['0', 'windchill', 'gustspeed', 'heatindex', 'events', 'precip', 'time', 'date'], axis=1, inplace=True)


# In[7]:

data.tail(10)


# ### Now let's check the distribution of all numeric variables

# In[8]:

plt.hist(data['temp'].values, bins=25)
plt.show()


# ### I believe the temperature is in Fahrenheit with the range around (30, 105) and median around 50. It seems reasonable
# 

# In[9]:

plt.hist(data['pressure'].dropna().values, bins=25)
plt.show()


# ### Unit here is in 'inhg' seems normally distributed

# In[10]:

plt.hist(data['dewpoint'].dropna().values, bins=50)
plt.show()


# ### Dew point is the temperature to which air must be cooled to become saturated with water vapour. The values we have is in Fahrenhite. Following is the perception (from Wikipedia):
# 
# <pre>
# '''
# Dew point 	            Human perception	                                         Relative humidity at 32 °C (90 °F)
# gt 26 °C 	gt 80 °F 	Severely high, even deadly for asthma related illnesses 	 73% and higher
# 24–26 °C 	75–80 °F 	Extremely uncomfortable, fairly oppressive 	                 62–72%
# 21–24 °C 	70–74 °F 	Very humid, quite uncomfortable 	                         52–61%
# 18–21 °C 	65–69 °F 	Somewhat uncomfortable for most people at upper edge 	     44–51%
# 16–18 °C 	60–64 °F 	OK for most, but all perceive the humidity at upper edge 	 37–43%
# 13–16 °C 	55–59 °F 	Comfortable 	                                             31–36%
# 10–12 °C 	50–54 °F 	Very comfortable 	                                         26–30%
# ls 10 °C 	ls 50 °F 	A bit dry for some 	                                         25% and lower
# 
# '''
# </pre>
# 
# ### The distribution seems reasonable but there is no value in 50 F bucket. This could be the sign of error in recording equipment.

# In[11]:

plt.hist(data['humidity'].dropna().values, bins=25)
plt.show()


# #### Humidity is in % bounded by (0,100). Distribution seems normal

# In[12]:

plt.hist(data['windspeed'].dropna().values, bins=25)
plt.show()


# #### I believe windspeed is in mph, distribution seems reasonable with right tail of windy days

# # Treating Missing values
# ### Now let's look the pattern in missing values. Based on pattern and understanding we will try to impute

# In[13]:

data.apply(lambda x: sum(x.isnull()),axis=0) 


# #### The categorical variables are mostly missing. But we can treat missing values of pressure and windspeed.
# #### Let's look at the pattern missing values.
# 
# #### Below I am plotting the occurrence of missing value with time

# In[14]:


plt.hist(pd.isnull(data['windspeed']).nonzero()[0], bins=1000)
plt.show()


# In[15]:

plt.hist(pd.isnull(data['pressure']).nonzero()[0], bins=1000)
plt.show()


# #### We can see the missing values of windspeed is evenly distributed but pressure is missing for a continuous span of time
# 
# #### I think the best way to impute windspeed is interpolating. For example, if windspeed for 1:00 AM is 6 mph and 3:00 AM is 7mph, we are imputing 6.5 mph for 2:00 AM
# 
# #### For pressure, we are using 2 methods for imputation. One is interpolation for few distributed missing values and mean for the missing time span

# In[16]:

data['pressure'] = data['pressure'].interpolate(limit=2, limit_direction='both')


# In[17]:

data['windspeed'] = data['windspeed'].interpolate(limit=4, limit_direction='both')


# In[18]:

data['pressure'].fillna(data['pressure'].mean(), inplace=True)


# In[19]:

data.apply(lambda x: sum(x.isnull()),axis=0) 


# # Bivariate Analysis

# In[20]:

from pandas.tools.plotting import scatter_matrix
scatter_matrix(data.drop(['conditions', 'winddir', 'visibility'], axis=1), alpha=0.2, figsize=(15, 15), diagonal='kde')
plt.show()


# ### temp-humidity has negative correlation whereas dewpoint has a linear threshold function of temp

# In[21]:

data.boxplot('windspeed', by = 'winddir', figsize=(12, 12))
plt.show()


# ### Looks like wind speed is very less in east direction. It is comparable with 'calm'

# ## Now let's see the historical pattern of temperature

# In[22]:

data['temp'].plot(figsize=(30, 6))
plt.show()


# In[23]:

ts = data['temp']
ts = ts.resample('W').mean()


# In[24]:

from pylab import rcParams
rcParams['figure.figsize'] = 30, 18

decomposition = sm.tsa.seasonal_decompose(ts, model='additive', freq=52)
fig = decomposition.plot()
plt.show()


# ### We can see clear yearly seasonality in temperature. There is no trend by year

# ## For this iteration, we will use seasonal ARIMA(p,d,q)(P,D,Q)s time series model. This model only considers the pattern of one variable.
# 
# ### There are total 6 hyperparameters to train. We will use the brute force grid search to find optimum parameter. We will AIC as our judging criteria
# 
# '''
# 
#     p: is the auto-regressive part of the model. It allows us to incorporate the effect of past values into our model. Intuitively, this would be similar to stating that it is likely to be warm tomorrow if it has been warm the past 3 days.
#     
#     d: is the integrated part of the model. This includes terms in the model that incorporate the amount of differencing (i.e. the number of past time points to subtract from the current value) to apply to the time series. Intuitively, this would be similar to stating that it is likely to be the same temperature tomorrow if the difference in temperature in the last three days has been very small.
#     
#     q: is the moving average part of the model. This allows us to set the error of our model as a linear combination of the error values observed at previous time points in the past.
#     
#     s: is the seasonality parameter. and P Q D is same as mentioned above but for seasonal component
# 
# '''

# In[25]:

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[26]:

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(ts,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# ### AIC is minimum (for non-zero p) for (1, 1, 1)x(1, 1, 1, 52) so we will use this parameter in our model. Grid search range can be further increased if there will be some pattern in residuals.

# In[27]:

mod = sm.tsa.statespace.SARIMAX(ts,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 52),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])


# ### The coef column shows the weight (i.e. importance) of each feature and how each one impacts the time series. The P>|z| column informs us of the significance of each feature weight. Here, each weight has a p-value lower or close to zero, so it is reasonable to retain all of them in our model.

# In[28]:

results.plot_diagnostics(figsize=(15, 12))
plt.show()


# ### The above charts are useful in understanding patterns in residual. It is desirable to have residuals uncorrelated and normally distributed with zero-mean.
# 
# #### All 4 charts are suggesting that residuals are white noise. If we find any pattern that means we can further improve model by expanding the grid search 

# # Validation
# 
# ### We will validate our model with 2 approaches. First with one step ahead forecast and second with dynamic forecast.

# In[29]:

pred = results.get_prediction(start=pd.to_datetime('2015-01-04'), dynamic=False)
pred_ci = pred.conf_int()


# In[30]:

ax = ts['2006':].plot(label='Observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

plt.legend()

plt.show()


# In[31]:

y_forecasted = pred.predicted_mean
y_truth = ts['2015-01-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# In[32]:

pred_dynamic = results.get_prediction(start=pd.to_datetime('2015-01-04'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()


# In[33]:

ax = ts['2006':].plot(label='Observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2015-01-01'), ts.index[-1],
                 alpha=.1, zorder=-1)
ax.set_xlabel('Date')
ax.set_ylabel('Temperature')


plt.legend()
plt.show()


# # Forecast
# 

# In[34]:

pred_uc = results.get_forecast(steps=150)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()


# In[35]:

ax = ts.plot(label='Observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Temperature')

plt.legend()
plt.show()


# # Next Steps
# 
# ## The model we have tried only consider the pattern of one variable. We can refine our model in the following ways:
# 
# ### 1. We can try VAR(Vector autoregression) model to capture the patterns of all 5 numeric variables in forecast of all others.
# ### 2. We can do feature engineering in all variables like '5 days moving avg temp', ' 20 days moving avg pressure' etc. and build regression model for one time step ahead
# ### 3. We can use a recurrent neural network to learn on sequential data and forecast.
# 
# ## I believe RNN can outperform others because it can easily learn seasonality and can do feature engineering. Also, ARIMA/VAR model struggles with more complexity and more data.

# In[ ]:




# In[ ]:



