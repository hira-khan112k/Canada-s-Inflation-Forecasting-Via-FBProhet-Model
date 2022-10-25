# Canada-s-Inflation-Forecasting-Via-FBProhet-Model
Forecasted Inflation Rate (CPI) For Canada with Machine Learning Model and used FBProphet Library
#Fb prophet
# load dataset
import pandas as pd
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from fbprophet import Prophet
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas.plotting import autocorrelation_plot
ratio=0.70
def parser(x):
	return datetime.strptime('196'+x, '%y-%m')
series = read_csv('C:/python/inflation_Canada.csv', parse_dates=["DATE"],index_col='DATE')

# split into train and test sets
X1 = series.values
size1 = int(len(X1) * ratio)
train1, test1 = X1[:size1], X1[size1:len(X1)]
df = pd.read_csv('C:/python/inflation_Canada.csv', index_col='DATE', parse_dates=True) # upload data file
df.head()
df = df.reset_index()
df.head()

series=df.rename(columns={'DATE':'ds', 'Canada':'y'})# chiang column headings into ds and y

# split into train and test sets
X = series
size = int(len(X) * ratio)
train, test = X[:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

# walk-forward validation
predictionsFbP = list()

for t in range(len(test)):
	model = Prophet(yearly_seasonality=False) # yearly seasnality false since data is monthly 
	model.fit(train);
	output = model.predict(test)
	yhat = output['yhat'][t]
	predictionsFbP.append(yhat)
	obs = test.loc[[t+len(train)]]
	train.append(obs)
	print('Month=%d, Predicted=%f, Expected=%f'% (t+1, yhat, test1[t])) #('predicted=%f, expected=%f' % (yhat, obs))

# evaluate forecasts
rmseFbP = sqrt(mean_squared_error(test1, predictionsFbP))
print('Test RMSE: %.3f' % rmseFbP)
r2FbP=r2_score(test1, predictionsFbP)
maeFbP = mean_absolute_error(test1, predictionsFbP)
mapeFbP = mean_absolute_percentage_error(test1, predictionsFbP)
#print(r2_score(test, predictions))
print('Test R2: %.3f' %r2FbP)
#plot forecasts against actual outcomes
def cm_to_inch(value):
    return value/2.54
plt.figure(figsize=(cm_to_inch(25), cm_to_inch(15)))
pyplot.plot(test1)
pyplot.plot(predictionsFbP, color='red')
plt.title('Inflation Forecasting through FB Prophet', fontsize=10)
plt.legend(('Test 20%', 'Predictions'), fontsize=10)
plt.xlabel('Number of Observations')
plt.ylabel('Inflation Rate')
pyplot.show()
