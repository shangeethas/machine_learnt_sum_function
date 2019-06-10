# libraries
import h5py
import pandas as pd
import os.path
import yaml as yaml


with open("configs_validate.yaml", 'r') as stream:
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
print("Testing Data Shape : ", df.shape)

h5f = h5py.File('linear_reg_model.hdf5', 'r')
intercept = h5f['intercept'][()]
coefficients = h5f['coefficients'][()]
h5f.close()

h5f = h5py.File('neural_networks_model.hdf5', 'r')
h5f.close()

h5f.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
h5f.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

Y_Model = []
A = df['a']
B = df['b']
C = df['c']
D = df['d']
Y = df['y']
i = 1
sum_error = 0.0
sum_score = 0.0

for i in range(df['a'].count()-1):
    y_model = coefficients[0] * A[i] + coefficients[1] * B[i] + \
              coefficients[2] * C[i] + coefficients[3] * D[i]
    y_actual = Y[i]

    error = y_model - y_actual
    sum_error += (error ** 2)

    score = h5f.evaluate((A[i], B[i], C[i], D[i]), Y[i], verbose=0)
    sum_score += score

print('Mean Sum Error of Linear Regression Model : ', sum_error / df['a'].count())
print('Sum Score of Neural Networks Model : ', sum_score)






