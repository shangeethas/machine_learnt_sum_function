# Take input data and learn a model (linear regression, neural network)
# Save the model in HDF5 format

# libraries
import os.path
import pandas as pd
from sklearn import linear_model
import h5py
import numpy as np


print('We are generating a model with training data set')
directory_name = input('Enter training data set directory name :')
if not os.path.realpath(directory_name):
    print('Invalid directory path specified')
    exit(4)

file_name = input('Enter training data set file name : ')
if not os.path.isfile(file_name):
    print('Invalid file name specified')
    exit(5)

file_str = os.path.join(directory_name, file_name)
print('Generated file string : ', file_str)

df = pd.read_csv(file_str)
print("Training Data Shape : ", df.shape)

d = {'a': df['a'], 'b': df['b'], 'c': df['c']}
X = pd.DataFrame(data=d)

Y = df['y']
reg = linear_model.LinearRegression()
reg.fit(X, Y)
intercept = reg.intercept_
coefficients = reg.coef_


h5f = h5py.File('model.hdf5', 'w')
h5f.create_dataset('intercept', data=np.array(intercept))
h5f.create_dataset('coefficients', data=np.array(coefficients))
h5f.close()
