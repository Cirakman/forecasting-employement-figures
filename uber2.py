from math import sqrt
import numpy as np
import os.path
import numpy as np
from numpy import genfromtxt
import pandas as pd
from keras.layers import Dense, GaussianNoise
from keras.layers import LSTM
from keras.layers import Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras import regularizers
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import math
import tensorflow as tf
from sklearn import preprocessing
import time
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift, 
                                         mean, 
                                         seasonal_naive)
tf.logging.set_verbosity(tf.logging.ERROR)

DATASET_STATES_CSV = 'States_Timeseries.csv'
FORECASTING_STEPS = 6
RANDOM_SEED = 3

SCALER_STANDARD = preprocessing.StandardScaler
SCALER_MINMAX = preprocessing.MinMaxScaler
DATA_SCALER = SCALER_MINMAX


def reshapeInputs(data):
    data = np.array(data)
    return np.reshape(data, (data.shape[0], data.shape[1],1))

def normalize_df(df,scaler):
    x = df.values #returns a numpy array
    x_scaled = scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)
    return df

def normalizeInputTouple_uber(x,y,offset):
    x = np.array(x)
    x -= offset
    y = np.array(y)
    y -= offset
    return x,y

def getTrainTestSplit(series):
    series_np = np.array(series)
    return series_np[:-FORECASTING_STEPS],series_np[-FORECASTING_STEPS:]


def create_dataset_uberSc(dataset,trends,season,LOOK_BACK):
    dataX, dataY, offsets,x_seas,y_seas= [], [],[],[],[]
    trends = np.array(trends)
    dataset = np.array(dataset)
    seas = np.array(season)
    for i in range(0,len(dataset) - LOOK_BACK -FORECASTING_STEPS+1):
        x = np.copy(dataset[i:(i + LOOK_BACK)])
        x_s = np.copy(seas[i:(i + LOOK_BACK)])
        y = np.copy(dataset[i + LOOK_BACK:i+LOOK_BACK+FORECASTING_STEPS])
        y_s = np.copy(seas[i + LOOK_BACK:i+LOOK_BACK+FORECASTING_STEPS])
        offset = trends[i+LOOK_BACK-1]
        x,y = normalizeInputTouple_uber(x,y,offset)
        dataX.append(x)
        dataY.append(y)
        offsets.append(offset)
        x_seas.append(x_s)
        y_seas.append(y_s)
    return reshapeInputs(dataX), np.array(dataY),np.array(offsets),x_seas,y_seas

def createDataset_uber(look_back):
    scaler = preprocessing.MinMaxScaler()
    df = pd.read_csv(DATASET_STATES_CSV,index_col=0,parse_dates=True,encoding = "utf-8")
    df = normalize_df(df,DATA_SCALER())
    df +=1
    df  = df.apply(np.log)
    df2 = df.copy()
    for col in df:
        decomp = decompose(df[col], period=12)
        df[col]= decomp.trend + decomp.resid
        df2[col+'_t'] = decomp.trend
        df2[col+'_s'] = decomp.seasonal
        df2[col+'_r'] = decomp.resid
    res = []
    for col in df:
        x,y,offsets,x_s,y_s = create_dataset_uberSc(df[col],df2[col+'_t'],df2[col+'_s'],look_back)
        x_test =np.copy(x[-FORECASTING_STEPS:])
        y_test = np.copy(y[-FORECASTING_STEPS:])
        offsets_test = np.copy(offsets[-FORECASTING_STEPS:])
        x_s_test = np.copy(x_s[-FORECASTING_STEPS:])
        y_s_test = np.copy(y_s[-FORECASTING_STEPS:])
        x = x[:-FORECASTING_STEPS]
        y = y[:-FORECASTING_STEPS]
        x_s = x_s[:-FORECASTING_STEPS]
        y_s = y_s[:-FORECASTING_STEPS]
        offsets = offsets[:-FORECASTING_STEPS]
        x_val = np.copy(x[-1:])
        y_val = np.copy(y[-1:])
        x_s_val = np.copy(x_s[-1:])
        y_s_val = np.copy(y_s[-1:])
        offsets_val = np.copy(offsets[-1:])
        x_train = np.copy(x[:-1])
        y_train = np.copy(y[:-1])
        x_s_train = np.copy(x_s[:-1])
        y_s_train = np.copy(y_s[:-1])
        offsets_train = np.copy(offsets[:-1])
        #y_test = test
        #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=RANDOM_SEED)
        res.append((x_train, x_val,x_test, y_train, y_val,y_test,offsets_train,offsets_val,offsets_test,x_s_train, x_s_val,x_s_test, y_s_train, y_s_val,y_s_test))
    return res

def rescale_uber(data,offset,seasonal):
    data = np.array(data)
    data += offset
    data += seasonal[0]
    data = np.exp(data)
    data -= 1
    return data

def evaluateUber(pred,y,y_s,o):
    pred = rescale_uber(pred,o,y_s)
    y = rescale_uber(y,o,y_s)
    score = mean_squared_error(y.ravel(),pred.ravel())
    return score 

def getModel_old(config):
    window = 15
    model = Sequential()
    stddev = config['noise'][0]
    model.add(GaussianNoise(stddev,batch_input_shape=(config['batchsize'][0],window, 1)))
    model.add(LSTM(units=config['cs1'][0],stateful=True, activation='tanh',kernel_regularizer=regularizers.l2(config['l2'][0])))
    model.add(Dense(FORECASTING_STEPS, activation='sigmoid'))
    lr = config['lr'][0]
    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mse', optimizer=opt,  metrics=['mean_absolute_error'])
    return model

def getModel(config):
    window = config['window'][0]
    model = Sequential()
    stddev = config['noise'][0]
    model.add(GaussianNoise(stddev,batch_input_shape=(config['batchsize'][0],window, 1)))
    model.add(LSTM(config['cs1'][0], activation='tanh',stateful=True,return_sequences = True))
    model.add(LSTM(config['cs2'][0], stateful = True,activation='tanh'))
    model.add(Dense(FORECASTING_STEPS, activation='sigmoid'))
    lr = math.pow(10,-config['lr'][0])
    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mse', optimizer=opt,  metrics=['mean_absolute_error'])
    return model

def getModel_bs(config):
    window = config['window'][0]
    model = Sequential()
    stddev = config['noise'][0]
    model.add(GaussianNoise(stddev,batch_input_shape=(1,window, 1)))
    model.add(LSTM(units=config['cs1'][0],stateful=True, activation='tanh',kernel_regularizer=regularizers.l2(config['l2'][0])))
    model.add(Dense(FORECASTING_STEPS, activation='sigmoid'))
    lr = config['lr'][0]
    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mse', optimizer=opt,  metrics=['mean_absolute_error'])
    return model

def uber_adjust_batchsize(model,config):
    new_model = getModel_bs(config)
    old_weights = model.get_weights()
    new_model.set_weights(old_weights)
    return new_model

def evaluateModel(config,rep):
    results = []
    config['epochs']=[50]
    config['batchsize']=[1]
    look_back = config['window'][0]
    epochs = config['epochs'][0]
    batchsize = config['batchsize'][0]
    data = createDataset_uber(look_back)
    #model.save_weights('.rand.h5')
    score = 0
    state = 0
    for x_t,x_v,x_te,y_t,y_v,y_te,o_t,o_v,o_te,x_s_t,x_s_v,x_s_te,y_s_t,y_s_v,y_s_te in data:
        model = getModel(config)
        #model.load_weights('.rand.h5')
        x_t = np.vstack((x_t,x_v))#add val to train
        y_t = np.vstack((y_t,y_v))
        if os.path.isfile('Models/uber2_'+str(state)+'_'+str(rep)+'.h5'):
            model = load_model('Models/uber2_'+str(state)+'_'+str(rep)+'.h5')
        else:
            for i in range(epochs):
                model.reset_states()
                model.fit(x_t, y_t, epochs=1, batch_size=batchsize,shuffle = False, verbose=0)
            model.save('Models/uber2_'+str(state)+'_'+str(rep)+'.h5')
        #model = uber_adjust_batchsize(model,config)
        model.reset_states()

        state = state+1
        model.predict(x_t,batch_size = 1)
        res = model.predict(x_te,batch_size = 1)[-1]
        res = evaluateUber(res,y_te[-1],y_s_te[-1],o_te[-1])
        #print(res)
        results.append(res)
        score = score + res
    score = score/len(data)
    K.clear_session()
    return results

