# libraries
import h5py
import pandas as pd
import os.path
import yaml as yaml


def load_test_data():
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
    return df


def validate_regression_model(df):
    h5f = h5py.File('linear_reg_model.hdf5', 'r')
    intercept = h5f['intercept'][()]
    coefficients = h5f['coefficients'][()]
    h5f.close()

    sum_error = 0.0
    for i in range(df['a'].count()-1):
        y_model = coefficients[0] * df['a'][i] + coefficients[1] * df['b'][i] + \
                  coefficients[2] * df['c'][i] + coefficients[3] * df['d'][i] + intercept
        y_actual = df['y'][i]

        error = y_model - y_actual
        sum_error += (error ** 2)

    print('Mean Sum Error of Linear Regression Model : ', sum_error / df['a'].count())


def validate_neural_networks_model(df):
    h5f = h5py.File('neural_networks_model.hdf5', 'r')
    h5f.close()

    h5f.load_weights("model.h5")
    print("Loaded model from disk")

    h5f.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    sum_score = 0.0

    for i in range(df['a'].count()-1):
        score = h5f.evaluate((df['a'][i], df['b'][i], df['c'][i], df['d'][i]), df['y'][i], verbose=0)
        sum_score += score

    print('Sum Score of Neural Networks Model : ', sum_score)


if __name__ == '__main__':
    test_df = load_test_data()
    validate_regression_model(test_df)
    # validate_neural_networks_model(df)




