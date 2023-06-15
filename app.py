import streamlit as st
import pandas as pd
import numpy as np
import math
import datetime as dt
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from numpy import array
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

global stock_dataset

st.set_page_config(page_title="Quantitative Algorithms Stock", page_icon="💹", layout="wide")

with st.container():
    st.title("Stock Price Detection using Quantitative Algorithms")

header = st.container()
database = st.container()
initial = st.container()
body = st.container()
svr = st.container()
random_forest = st.container()
knn = st.container()
lstm = st.container()
gru = st.container()
hybrid = st.container()
final = st.container()


with header:
    st.subheader("DataSet Upload")
    data_file=st.file_uploader("Upload Stock Dataset in CSV",type=["csv"])
    def fileCheck():
        if data_file is not None:
            stock_dataset=pd.read_csv(data_file)
            return stock_dataset


stock_dataset = fileCheck()
if(stock_dataset is None):
    time.sleep(1000)

with database:
        st.dataframe(stock_dataset)
global bist100
bist100 = stock_dataset
bist100.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close"}, inplace= True)
# Checking null value
bist100.isnull().sum()
# Convert date field from string to Date format and make it index
# Sort Values by Date
bist100['date'] = pd.to_datetime(bist100.date)
bist100.sort_values(by='date', inplace=True)
df2 = bist100[bist100.date>='2015-01-01']
df2 = df2[df2.date<'2020-01-01']
bist100=df2

with initial:
    st.write("Starting date: ",bist100.iloc[0][0])
    st.write("Ending date: ", bist100.iloc[-1][0])
    st.write("Duration: ", bist100.iloc[-1][0]-bist100.iloc[0][0])

closedf = bist100[['date','close']]

with body:  
    fig = px.line(closedf, x=closedf.date, y=closedf.close,labels={'date':'Date Time Range','close':'Close Stock Price (in Rs)'})
    fig.update_traces(marker_line_width=2, opacity=1)
    fig.update_layout(title_text='Stock Closing Price Chart', plot_bgcolor='white', font_size=15, font_color='black')
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    st.plotly_chart(fig, use_container_width=True)


close_stock = closedf.copy()
del closedf['date']
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
training_size=int(len(closedf)*0.70)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]

def create_dataset(dataset, time_step=3):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)
time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


#graph = [['Algorithms','Main Tuning Parameters (Hyper parameters)','RMSE','MSE','MAE','Variance Regressio Score','R**2 Score',
#            'Mean Gamma Deviance','Mean Poisson Deviance']]
#st.write(graph)

#Algorithms
### Super vector regression - SVR

from sklearn.svm import SVR

svr_rbf = SVR(kernel = 'rbf', C = 1e2, gamma = 0.01)
svr_rbf.fit(X_train, y_train,)
train_predict=svr_rbf.predict(X_train)
test_predict=svr_rbf.predict(X_test)

train_predict = train_predict.reshape(-1,1)
test_predict = test_predict.reshape(-1,1)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 


look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict

names = cycle(['Original close price','Train predicted close price','Test predicted close price'])

plotdf = pd.DataFrame({'date': close_stock['date'],
                       'original_close': close_stock['close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

lst_output=[]
n_steps=time_step
i=0
pred_days = 10
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        
        yhat = svr_rbf.predict(x_input)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat.tolist())
        temp_input=temp_input[1:]
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        yhat = svr_rbf.predict(x_input)
        
        temp_input.extend(yhat.tolist())
        lst_output.extend(yhat.tolist())
        
        i=i+1

last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)

temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

with svr:
    fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':'Stock price','date': 'Date'})
    fig.update_traces(marker_line_width=10, opacity=1)
    fig.update_layout(title_text='Support Vector Regression Model (SVR)',
                  plot_bgcolor='black', font_size=15, font_color='black',legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain,train_predict)))
    st.write("Train data MSE: ", mean_squared_error(original_ytrain,train_predict))
    st.write("Test data MAE: ", mean_absolute_error(original_ytrain,train_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))
    st.write("Test data MSE: ", mean_squared_error(original_ytest,test_predict))
    st.write("Test data MAE: ", mean_absolute_error(original_ytest,test_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Train data explained variance regression score:", explained_variance_score(original_ytrain, train_predict))
    st.write("Test data explained variance regression score:", explained_variance_score(original_ytest, test_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Train data MGD: ", mean_gamma_deviance(original_ytrain, train_predict))
    st.write("Test data MGD: ", mean_gamma_deviance(original_ytest, test_predict))
    st.write("----------------------------------------------------------------------")
    st.write("Train data MPD: ", mean_poisson_deviance(original_ytrain, train_predict))
    st.write("Test data MPD: ", mean_poisson_deviance(original_ytest, test_predict))

    st.write("Next 10 days Prediction Values from SVR are ",next_predicted_days_value[16:])



#Random Forest
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, y_train)


train_predict=regressor.predict(X_train)
test_predict=regressor.predict(X_test)

train_predict = train_predict.reshape(-1,1)
test_predict = test_predict.reshape(-1,1)


train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 

look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict

names = cycle(['Original close price','Train predicted close price','Test predicted close price'])


plotdf = pd.DataFrame({'date': close_stock['date'],
                       'original_close': close_stock['close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

lst_output=[]
n_steps=time_step
i=0
pred_days = 10
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        
        yhat = regressor.predict(x_input)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat.tolist())
        temp_input=temp_input[1:]
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        yhat = regressor.predict(x_input)
        
        temp_input.extend(yhat.tolist())
        lst_output.extend(yhat.tolist())
        
        i=i+1

last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)
temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

with random_forest:
    fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':'Stock price','date': 'Date'})
    fig.update_layout(title_text='Random Forest Regressor Model',
                  plot_bgcolor='black', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain,train_predict)))
    st.write("Train data MSE: ", mean_squared_error(original_ytrain,train_predict))
    st.write("Test data MAE: ", mean_absolute_error(original_ytrain,train_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))
    st.write("Test data MSE: ", mean_squared_error(original_ytest,test_predict))
    st.write("Test data MAE: ", mean_absolute_error(original_ytest,test_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Train data explained variance regression score:", explained_variance_score(original_ytrain, train_predict))
    st.write("Test data explained variance regression score:", explained_variance_score(original_ytest, test_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Train data MGD: ", mean_gamma_deviance(original_ytrain, train_predict))
    st.write("Test data MGD: ", mean_gamma_deviance(original_ytest, test_predict))
    st.write("----------------------------------------------------------------------")
    st.write("Train data MPD: ", mean_poisson_deviance(original_ytrain, train_predict))
    st.write("Test data MPD: ", mean_poisson_deviance(original_ytest, test_predict))

    st.write("Next 10 days Prediction Values from Random Forest Regresson are ",next_predicted_days_value[16:])




#K Nearest Neighbour
from sklearn import neighbors

K = time_step
neighbor = neighbors.KNeighborsRegressor(n_neighbors = K)
neighbor.fit(X_train, y_train)

train_predict=neighbor.predict(X_train)
test_predict=neighbor.predict(X_test)

train_predict = train_predict.reshape(-1,1)
test_predict = test_predict.reshape(-1,1)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))

look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict

names = cycle(['Original close price','Train predicted close price','Test predicted close price'])

plotdf = pd.DataFrame({'date': close_stock['date'],
                       'original_close': close_stock['close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

lst_output=[]
n_steps=time_step
i=0
pred_days = 10
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        
        yhat = neighbor.predict(x_input)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat.tolist())
        temp_input=temp_input[1:]
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        yhat = neighbor.predict(x_input)
        
        temp_input.extend(yhat.tolist())
        lst_output.extend(yhat.tolist())
        
        i=i+1

last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)
temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

with knn:
    fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':'Stock price','date': 'Date'})
    fig.update_layout(title_text='K Nearest Neighbour Model',
                  plot_bgcolor='black', font_size=15, font_color='black',legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain,train_predict)))
    st.write("Train data MSE: ", mean_squared_error(original_ytrain,train_predict))
    st.write("Test data MAE: ", mean_absolute_error(original_ytrain,train_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))
    st.write("Test data MSE: ", mean_squared_error(original_ytest,test_predict))
    st.write("Test data MAE: ", mean_absolute_error(original_ytest,test_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Train data explained variance regression score:", explained_variance_score(original_ytrain, train_predict))
    st.write("Test data explained variance regression score:", explained_variance_score(original_ytest, test_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Train data MGD: ", mean_gamma_deviance(original_ytrain, train_predict))
    st.write("Test data MGD: ", mean_gamma_deviance(original_ytest, test_predict))
    st.write("----------------------------------------------------------------------")
    st.write("Train data MPD: ", mean_poisson_deviance(original_ytrain, train_predict))
    st.write("Test data MPD: ", mean_poisson_deviance(original_ytest, test_predict))
    st.write("Next 10 days Prediction Values from KNN are ",next_predicted_days_value[16:])


#LSTM
from tensorflow.keras.layers import LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

tf.keras.backend.clear_session()
model=Sequential()
model.add(LSTM(32,return_sequences=True,input_shape=(time_step,1)))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=3,batch_size=5,verbose=1)

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
train_predict.shape, test_predict.shape

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 

look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict

names = cycle(['Original close price','Train predicted close price','Test predicted close price'])


plotdf = pd.DataFrame({'date': close_stock['date'],
                       'original_close': close_stock['close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 10
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1

last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)
temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]


with lstm:
    fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':'Stock price','date': 'Date'})
    fig.update_layout(title_text='Long Short Term Memory Model',
                  plot_bgcolor='black', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)    
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain,train_predict)))
    st.write("Train data MSE: ", mean_squared_error(original_ytrain,train_predict))
    st.write("Test data MAE: ", mean_absolute_error(original_ytrain,train_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))
    st.write("Test data MSE: ", mean_squared_error(original_ytest,test_predict))
    st.write("Test data MAE: ", mean_absolute_error(original_ytest,test_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Train data explained variance regression score:", explained_variance_score(original_ytrain, train_predict))
    st.write("Test data explained variance regression score:", explained_variance_score(original_ytest, test_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Train data MGD: ", mean_gamma_deviance(original_ytrain, train_predict))
    st.write("Test data MGD: ", mean_gamma_deviance(original_ytest, test_predict))
    st.write("----------------------------------------------------------------------")
    st.write("Train data MPD: ", mean_poisson_deviance(original_ytrain, train_predict))
    st.write("Test data MPD: ", mean_poisson_deviance(original_ytest, test_predict))
    st.write("Next 10 days Prediction Values from LSTM are ",next_predicted_days_value[16:])



#GRU
from tensorflow.keras.layers import GRU

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

tf.keras.backend.clear_session()
model=Sequential()
model.add(GRU(32,return_sequences=True,input_shape=(time_step,1)))
model.add(GRU(32,return_sequences=True))
model.add(GRU(32,return_sequences=True))
model.add(GRU(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=3,batch_size=5,verbose=1)

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))

look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict


names = cycle(['Original close price','Train predicted close price','Test predicted close price'])

plotdf = pd.DataFrame({'date': close_stock['date'],
                       'original_close': close_stock['close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 10
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
             
last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)
temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

with gru:
    fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':'Stock price','date': 'Date'})
    fig.update_layout(title_text='Gated Recurrent Unit Model',
                  plot_bgcolor='black', font_size=15, font_color='black',legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain,train_predict)))
    st.write("Train data MSE: ", mean_squared_error(original_ytrain,train_predict))
    st.write("Test data MAE: ", mean_absolute_error(original_ytrain,train_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))
    st.write("Test data MSE: ", mean_squared_error(original_ytest,test_predict))
    st.write("Test data MAE: ", mean_absolute_error(original_ytest,test_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Train data explained variance regression score:", explained_variance_score(original_ytrain, train_predict))
    st.write("Test data explained variance regression score:", explained_variance_score(original_ytest, test_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Train data MGD: ", mean_gamma_deviance(original_ytrain, train_predict))
    st.write("Test data MGD: ", mean_gamma_deviance(original_ytest, test_predict))
    st.write("----------------------------------------------------------------------")
    st.write("Train data MPD: ", mean_poisson_deviance(original_ytrain, train_predict))
    st.write("Test data MPD: ", mean_poisson_deviance(original_ytest, test_predict))
    st.write("Next 10 days Prediction Values from GRU are ",next_predicted_days_value[16:])




#LSTM + GRU (Hybrid Model)

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

tf.keras.backend.clear_session()
model=Sequential()
model.add(LSTM(32,return_sequences=True,input_shape=(time_step,1)))
model.add(LSTM(32,return_sequences=True))
model.add(GRU(32,return_sequences=True))
model.add(GRU(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

model.summary()

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=3,batch_size=5,verbose=1)

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 

look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
print("Train predicted data: ", trainPredictPlot.shape)

# shift test predictions for plotting
testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
print("Test predicted data: ", testPredictPlot.shape)

names = cycle(['Original close price','Train predicted close price','Test predicted close price'])

plotdf = pd.DataFrame({'date': close_stock['date'],
                       'original_close': close_stock['close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 10
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)
temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

with hybrid:
    fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                            plotdf['test_predicted_close']],
                labels={'value':'Stock price','date': 'Date'})
    fig.update_layout(title_text='LSTM + GRU Hybrid Model',
                    plot_bgcolor='black', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain,train_predict)))
    st.write("Train data MSE: ", mean_squared_error(original_ytrain,train_predict))
    st.write("Test data MAE: ", mean_absolute_error(original_ytrain,train_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))
    st.write("Test data MSE: ", mean_squared_error(original_ytest,test_predict))
    st.write("Test data MAE: ", mean_absolute_error(original_ytest,test_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Train data explained variance regression score:", explained_variance_score(original_ytrain, train_predict))
    st.write("Test data explained variance regression score:", explained_variance_score(original_ytest, test_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Train data MGD: ", mean_gamma_deviance(original_ytrain, train_predict))
    st.write("Test data MGD: ", mean_gamma_deviance(original_ytest, test_predict))
    st.write("----------------------------------------------------------------------")
    st.write("Train data MPD: ", mean_poisson_deviance(original_ytrain, train_predict))
    st.write("Test data MPD: ", mean_poisson_deviance(original_ytest, test_predict))
    st.write("Next 10 days Prediction Values from Hybrid Model are ",next_predicted_days_value[16:])

# finaldf = pd.DataFrame({
#     'svr':svrdf,
#     'rf':rfdf,
#     'knn':knndf,
#     'lstm':lstmdf,
#     'gru':grudf,
#     'lstm_gru':lstmgrudf,
# })
# finaldf.head()


# names = cycle(['SVR', 'RF','KNN','LSTM','GRU','LSTM + GRU'])

# fig = px.line(finaldf[225:], x=finaldf.index[225:], y=[finaldf['svr'][225:],finaldf['rf'][225:], finaldf['knn'][225:], 
#                                           finaldf['lstm'][225:], finaldf['gru'][225:], finaldf['lstm_gru'][225:]],
#              labels={'x': 'Timestamp','value':'Stock close price'})
# fig.update_layout(title_text='Final stock analysis chart', font_size=15, font_color='black',legend_title_text='Algorithms')
# fig.for_each_trace(lambda t:  t.update(name = next(names)))
# fig.update_xaxes(showgrid=False)
# fig.update_yaxes(showgrid=False)

# fig.show()
