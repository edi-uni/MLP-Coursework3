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
from keras import backend as K

filename= "data.csv"
# raw = pd.read_csv(filename)
raw, unsorted_raw = kb.process_data()
#train, train_y, test, test_y = kb.split_data(raw, df, indexOfNull)
shot_zone_area_dict = {'Center(C)': 1, 'Left Side(L)': 4, 'Right Side(R)': 4, 'Left Side Center(LC)': 7, 'Right Side Center(RC)': 7, 'Back Court(BC)': 10}

# convert an array of values into a dataset matrix
'''
def create_dataset(train,train_y, look_back=1):
	dataX, dataY = [], []
	#train, train_y, test, test_y = kb.split_data(raw, dataset, indexOfNull)
	dataX = train.to_numpy()
	dataY = train_y.to_numpy()
	return numpy.array(dataX), numpy.array(dataY)

def create_dataset_test(test,test_y, look_back=1):
	dataX, dataY = [], []
	#train, train_y, test, test_y = kb.split_data(raw, dataset, indexOfNull)
	dataX = test.to_numpy()
	dataY = test_y.to_numpy()
	return numpy.array(dataX), numpy.array(dataY)
'''
numpy.random.seed(7)

raw, unsorted_raw = kb.process_data()
seasons_dict, seasons_loc_dict, seasons_mode_dict = kb.split_for_experiments(raw)
seasons_blocks_dict = kb.split_in_blocks(seasons_dict, 'season')
all_points = kb.get_all_testing_points(unsorted_raw)

def season_split():
    pd.options.mode.chained_assignment = None
    temp_block = pd.DataFrame()
    for k,v in seasons_blocks_dict.items():
        copy = v.copy()
        for i in range(len(copy)):
            print("For seasons value" ,numpy.unique(k),"the BLOCK is", i)
            temp_train, temp_train_y = kb.split_by_field(temp_block)
            train, train_y, test, test_y = kb.split_data(copy[i], all_points, 'season')
            if not test.empty:
                train = pd.concat([temp_train, train])
                train_y = pd.concat([temp_train_y, train_y])
                print("started rnn execution")
                K.clear_session()
                print("Session cleared. Starting training")
                history,testPredict = rnn(train, train_y, test, test_y)	
                print("Finished rnn execution")

            '''
            if len(test_y) > 0 :
                print("I reached here1")
                train = pd.concat([temp_train, train])
                train_y = pd.concat([temp_train_y, train_y])
                print("i reached here2")
                history,testPredict = rnn(train, train_y, test, test_y)	
                print("I left here 1")	
            else:
                print("I reached here3")
                train = pd.concat([temp_train, train])
                train_y = pd.concat([temp_train_y, train_y])
                print("I left this place")
            '''
            for j in test.index:
                copy[i].xs(j)['shot_made_flag'] = testPredict
            copy[i] = copy[i].drop('season', 1)
            temp_block = pd.concat([temp_block, copy[i]])
    py_plot(history)
        # break



def rnn(train, train_y, test, test_y):
	# fix random seed for reproducibility
	#numpy.random.seed(7)
	# load the dataset
	#dataframe = read_csv('data.csv', usecols=[1], engine='python', skipfooter=3)
	#raw,dataframe = kb.process_data()
	#train, train_y, test, test_y = kb.split_data(raw, dataframe, indexOfNull)
	#train, train_y, test, test_y = season_split()
	#dataset = dataframe.values
	#dataset = dataset.astype('float32')
	# normalize the dataset
	#scaler = MinMaxScaler(feature_range=(0, 1))
	#dataset = scaler.fit_transform(dataset)
	#scaler.min_,scaler.scale_=scaler.min_[0],scaler.scale_[0]
	# split into train and test sets
	#train_size = int(len(train))
	#test_size = len(test) 

	#train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	#train, train_y, test, test_y = kb.split_data(raw, dataframe, indexOfNull)
	dataset = train.values
	dataset = train.astype('float64')
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)
	scaler.min_,scaler.scale_=scaler.min_[0],scaler.scale_[0]
	# split into train and test sets
	train_size = int(len(train))
	test_size = len(test) 

	# reshape into X=t and Y=t+1
	look_back = train.shape[1]

	trainX, trainY = numpy.array(train.to_numpy()), numpy.array(train_y.to_numpy())
	testX, testY = numpy.array(test.to_numpy()), numpy.array(test_y.to_numpy())

	# reshape input to be [samples, time steps, features]
	trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, input_shape=(1, look_back),activation='tanh',dropout=0.25, use_bias = True))
	#model.add(LSTM(3, input_shape=(1, look_back),activation='sigmoid'))
	model.add(Dense(1))
	#model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['accuracy'])
	model.compile(loss='binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
	history = model.fit(trainX, trainY, epochs=100, batch_size=100, verbose=1,validation_data=(testX,testY))

	# make predictions
	trainPredict = model.predict_classes(trainX)
	testPredict = model.predict_classes(testX)
	'''
	print("train predict is",trainPredict.shape)
	print("test predict is",testPredict.shape)

	print("The train data shape is", train.shape)
	print("The train actual shape is", trainY.shape)

	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform(trainY)
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform(testY)

	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	'''
	return history,testPredict

def py_plot(history):
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

if __name__ == '__main__':
	season_split()
	
