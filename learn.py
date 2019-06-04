# Take input data and learn a model (linear regression, neural network)
# Save the model in HDF5 format

# libraries
import os.path
import pandas as pd
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense


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
print("Training Data Size : ", df.size)
print("Training Data Row Count : ", df['a'].count)

d = {'a': df['a'], 'b': df['b'], 'c': df['c']}
X = pd.DataFrame(data=d)

Y = df['y']
reg = linear_model.LinearRegression()
reg.fit(X, Y)
print('Intercept: \n', reg.intercept_)
print('Coefficients: \n', reg.coef_)


model = Sequential()
model.add(Dense(reg.intercept_, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(reg.coef_, kernel_initializer='uniform', activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")