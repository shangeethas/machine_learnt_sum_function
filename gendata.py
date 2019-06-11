# libraries
import numpy as np
import pandas as pd
import yaml as yaml


def gen_data():
    with open("configs_data.yaml", 'r') as stream:
        try:
            configs = yaml.safe_load(stream)
            print(configs)
        except yaml.YAMLError as exc:
            print(exc)

    random_seed_a = configs.get('random_seed_a')
    random_seed_b = configs.get('random_seed_b')
    random_seed_c = configs.get('random_seed_c')
    random_seed_d = configs.get('random_seed_d')
    number_points = configs.get('number_points')
    min_range = configs.get('min_range')
    max_range = configs.get('max_range')

    if min_range == max_range:
        print('Minimum and Maximum values of Range should be different')
        exit(1)
    elif min_range > max_range:
        print('Range Minimum should be less than Range Maximum value')
        exit(2)
    else:
        print('Valid inputs are provided by the user')

    np.random.seed(random_seed_a)
    random_integers_a = np.random.randint(min_range, max_range + 1, size=number_points)

    np.random.seed(random_seed_b)
    random_integers_b = np.random.randint(min_range, max_range + 1, size=number_points)

    np.random.seed(random_seed_c)
    random_integers_c = np.random.randint(min_range, max_range + 1, size=number_points)

    np.random.seed(random_seed_d)
    random_integers_d = np.random.randint(min_range, max_range + 1, size=number_points)

    output = []
    for i in range(number_points):
        # ground truth
        output.append(random_integers_a[i] + random_integers_b[i]
                      + random_integers_c[i] + random_integers_d[i])

    d = {'a': random_integers_a, 'b': random_integers_b,
         'c': random_integers_c, 'd': random_integers_d, 'y': output}
    df = pd.DataFrame(d)

    train_length = int(len(df) * 0.6)
    test_length = int(len(df) * 0.2)
    validate_length = int(len(df) * 0.2)

    df_train = df[0:train_length]
    print('Data Frame Train Shape : ', df_train.shape)

    df_test = df[train_length+1:train_length+test_length]
    print('Data Frame Test Shape : ', df_test.shape)

    df_validate = df[train_length+test_length+1:train_length+test_length+validate_length]
    print('Data Frame Validate Shape : ', df_validate.shape)

    file_train = open("numbers_train.csv", "w+")
    if file_train == "":
        print('File creation failed')
        exit(3)
    df_train.to_csv("numbers_train.csv", ",")
    corr = df_train.corr()
    print('correlation matrix for train data : \n', corr)

    file_test = open("numbers_test.csv", "w+")
    if file_test == "":
        print('File creation failed')
        exit(3)
    df_test.to_csv("numbers_test.csv", ",")
    corr = df_test.corr()
    print('correlation matrix for test data : \n', corr)

    file_validate = open("numbers_validate.csv", "w+")
    if file_validate == "":
        print('File creation failed')
        exit(3)
    df_validate.to_csv("numbers_validate.csv", ",")
    corr = df_validate.corr()
    print('correlation matrix for validate data : \n', corr)


if __name__ == '__main__':
    gen_data()

