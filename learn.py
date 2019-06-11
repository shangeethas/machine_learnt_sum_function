# libraries
import os.path
import pandas as pd
from sklearn import linear_model
import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import yaml as yaml


def load_train_data():
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
    return df


def linear_regression_learn(df):
    d = {'a': df['a'], 'b': df['b'], 'c': df['c'], 'd': df['d']}
    x = pd.DataFrame(data=d)

    y = df['y']
    reg = linear_model.LinearRegression()
    reg.fit(x, y)
    intercept = reg.intercept_
    coefficients = reg.coef_

    print('Intercept : ', intercept)
    print('Coefficients : ', coefficients)

    h5f = h5py.File('linear_reg_model.hdf5', 'w')
    h5f.create_dataset('intercept', data=np.array(intercept))
    h5f.create_dataset('coefficients', data=np.array(coefficients))
    h5f.close()


def neural_networks_learn():
    model = Sequential()
    model.add(Dense(1, kernel_initializer='uniform',
                    activation='relu', input_shape=X.shape))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    model.save('neural_networks_model.hdf5')


if __name__ == '__main__':
    train_df = load_train_data()
    linear_regression_learn(train_df)
    # neural_networks_learn()

