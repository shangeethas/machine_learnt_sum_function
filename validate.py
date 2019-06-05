# libraries
import h5py
import pandas as pd
import os.path

print('We are validating the model with test data set')

directory_name = input('Enter testing data set directory name :')
if not os.path.realpath(directory_name):
    print('Invalid directory path specified')
    exit(4)

file_name = input('Enter testing data set file name : ')
if not os.path.isfile(file_name):
    print('Invalid file name specified')
    exit(5)

file_str = os.path.join(directory_name, file_name)
print('Generated file string : ', file_str)

df = pd.read_csv(file_str)
print("Testing Data Shape : ", df.shape)

h5f = h5py.File('model.hdf5', 'r')
intercept = h5f['intercept'][()]
coefficients = h5f['coefficients'][()]
h5f.close()

Y_Model = []
A = df['a']
B = df['b']
C = df['c']
Y = df['y']
i = 1
sum_error = 0.0

for i in range(df['a'].count()-1):
    y_model = coefficients[0] * A[i] + coefficients[1] * B[i] + coefficients[2] * C[i]
    y_actual = Y[i]

    error = y_model - y_actual
    sum_error += (error ** 2)

print('Mean Sum Error : ', sum_error / df['a'].count())




