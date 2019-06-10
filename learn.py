# libraries
import os.path
import pandas as pd
from sklearn import linear_model
import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import yaml as yaml

with open("configs_learn.yaml", 'r') as stream:
    try:
        configs = yaml.safe_load(stream)
        print(configs)
    except yaml.YAMLError as exc:
        print(exc)

directory_name = configs.get('directory_name')
if not os.path.realpath(directory_name):
    print('Invalid directory path specified')
    exit(4)

file_name = configs.get('file_name')
if not os.path.isfile(file_name):
    print('Invalid file name specified')
    exit(5)

file_str = os.path.join(directory_name, file_name)
print('Generated file string : ', file_str)

df = pd.read_csv(file_str)
print("Training Data Shape : ", df.shape)

d = {'a': df['a'], 'b': df['b'], 'c': df['c'], 'd': df['d']}
X = pd.DataFrame(data=d)

Y = df['y']
reg = linear_model.LinearRegression()
reg.fit(X, Y)
intercept = reg.intercept_
coefficients = reg.coef_


h5f = h5py.File('linear_reg_model.hdf5', 'w')
h5f.create_dataset('intercept', data=np.array(intercept))
h5f.create_dataset('coefficients', data=np.array(coefficients))
h5f.close()

model = Sequential()
model.add(Dense(12, kernel_initializer='uniform',
                activation='relu', input_shape=X.shape))
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
h5f = h5py.File('neural_networks_model.hdf5', 'w')
h5f.close()