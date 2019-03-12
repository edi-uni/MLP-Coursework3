# LSTM for international airline passengers problem with window regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import kobe_bryant_classifier as kb

filename= "data.csv"
# raw = pd.read_csv(filename)
raw, df, indexOfNull = kb.process_data()
shot_zone_area_dict = {'Center(C)': 1, 'Left Side(L)': 4, 'Right Side(R)': 4, 'Left Side Center(LC)': 7, 'Right Side Center(RC)': 7, 'Back Court(BC)': 10}

def process_data():
    raw = pd.read_csv(filename)
    # drops = ['shot_id', 'team_id', 'team_name', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', \
             # 'matchup', 'game_event_id', 'game_id', 'game_date']
    # drops = ['team_id', 'team_name', 'matchup', 'game_id', 'game_date']
    drops = ['team_id', 'team_name', 'game_event_id', 'game_id', 'game_date', 'shot_id']
    for drop in drops:
        raw = raw.drop(drop, 1)

    # categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'season']
    # categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'season', 'shot_id', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', \
    #          'matchup', 'game_event_id', 'game_date']
    categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', 'opponent', 'season']
    for var in categorical_vars:
        raw = pd.concat([raw, pd.get_dummies(raw[var], prefix=var)], 1)
        raw = raw.drop(var, 1)

    for i, row in enumerate(raw.itertuples(), 1):
        if "@" in row.matchup:
            raw.set_value(row.Index, 'matchup', 0)
        else:
            raw.set_value(row.Index, 'matchup', 1)

    df = raw[pd.notnull(raw['shot_made_flag'])]
    indexOfNull = raw[raw['shot_made_flag'].isnull()].index.tolist()

    return raw, df, indexOfNull

def split_data(raw, df, indexOfNull):
    per_test =  round((15 * len(df))/100)
    n =[0 for i in range(per_test)]

    m = len(indexOfNull)-1
    for z in range(per_test):
        c = indexOfNull[m]+1
        x = raw.iloc[c]
        #print(pd.notnull(x['shot_made_flag']))
        flg = 0
        if(pd.isnull(x['shot_made_flag'])):
            #print(c)
            flg =1
            while flg==1:
                c = c+1
                x = raw.iloc[c]
                if pd.notnull(x['shot_made_flag']):
                    flg =0
        n[z] = c
        m=m-1;
    test_comp = raw.iloc[n]
    test = test_comp.drop('shot_made_flag', 1)
    test_y = test_comp['shot_made_flag']
    df = raw.drop(raw.index[n])
    df= df[pd.notnull(df['shot_made_flag'])]
    # separate df into explanatory and response variables
    train = df.drop('shot_made_flag', 1)
    train_y = df['shot_made_flag']
    return train, train_y, test, test_y

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	train, train_y, test, test_y = split_data(raw, dataset, indexOfNull)
	dataX = train.to_numpy()
	dataY = train_y.to_numpy()
	return numpy.array(dataX), numpy.array(dataY)

def create_dataset_test(dataset, look_back=1):
	dataX, dataY = [], []
	train, train_y, test, test_y = split_data(raw, dataset, indexOfNull)
	dataX = test.to_numpy()
	dataY = test_y.to_numpy()
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
#dataframe = read_csv('data.csv', usecols=[1], engine='python', skipfooter=3)
raw,dataframe,indexOfNull = process_data()
train, train_y, test, test_y = split_data(raw, dataframe, indexOfNull)
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
scaler.min_,scaler.scale_=scaler.min_[0],scaler.scale_[0]
# split into train and test sets
train_size = int(len(train))
test_size = len(test) 

#train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
train, train_y, test, test_y = split_data(raw, dataframe, indexOfNull)

# reshape into X=t and Y=t+1
look_back = train.shape[1]
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset_test(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(3, input_shape=(1, look_back),activation='tanh'))
#model.add(LSTM(3, input_shape=(1, look_back),activation='sigmoid'))
model.add(Dense(1))
#model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
history = model.fit(trainX, trainY, epochs=1000, batch_size=1000, verbose=1,validation_data=(testX,testY))

# make predictions
trainPredict = model.predict_classes(trainX)
testPredict = model.predict_classes(testX)
print("train predict is",trainPredict.shape)
print("test predict is",testPredict.shape)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

#Plotting the accuracy of the neural network
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy of the RNN')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('RNN Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

