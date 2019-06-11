# libraries
import h5py
import yaml as yaml


def predict():
    with open("configs_predict.yaml", 'r') as stream:
        try:
            configs = yaml.safe_load(stream)
            print(configs)
        except yaml.YAMLError as exc:
            print(exc)

    a = configs.get('a')
    b = configs.get('b')
    c = configs.get('c')
    d = configs.get('d')

    h5f = h5py.File('linear_reg_model.hdf5', 'r')
    intercept = h5f['intercept'][()]
    coefficients = h5f['coefficients'][()]
    h5f.close()

    y_actual = a + b + c + d
    print('Actual Sum Function of {} and {} and {} and {} is : {}'.format(a, b, c, d, y_actual))

    y_model = coefficients[0] * a + coefficients[1] * b + coefficients[2] * c + coefficients[3] * d + intercept
    print('Machine Learnt Sum Function of {} and {} and {} and {} is : {}'.format(a, b, c, d, y_model))


if __name__ == '__main__':
    predict()
