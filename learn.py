# Take input data and learn a model (linear regression, neural network)
# Save the model in HDF5 format

# libraries
import os.path
import pandas as pd
import numpy as np


class Model:
    def __init__(self, coefficient_i, coefficient_j, coefficient_k):
        print("This is the constructor method.")
        self.a = coefficient_i
        self.b = coefficient_j
        self.c = coefficient_k

    def get_sum(self, i, j, k):
        return self.a * i + self.b * j + self.c * k


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

error_sum = 0
model = Model(1, 1, 1)

for _ in range(df.size):
    # ground truth
    i = np.random.choice(df['0'])
    j = np.random.choice(df['0'])
    k = np.random.choice(df['0'])
    y_actual = i + j + k
    print('i : {}, j:{}, k:{}, y_actual:{}'.format(i, j, k, y_actual))

    y_model = model.get_sum(i, j, k)

    # sum of errors
    error = y_model - y_actual
    error_sum_square = error_sum + error ^ 2

mean_error_sum_square = error_sum_square / df.size
print('Mean Error Sum Square : {}'.format(mean_error_sum_square))

df_model = pd.DataFrame.from_records((1, 2, 3))
df_model.to_hdf('store_tl.h5', 'table', append=True)