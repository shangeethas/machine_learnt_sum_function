# libraries
import numpy as np
import pandas as pd
import yaml as yaml

with open("configs_test.yaml", 'r') as stream:
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
random_test_integers_a = np.random.randint(min_range, max_range + 1, size=number_points)

np.random.seed(random_seed_b)
random_test_integers_b = np.random.randint(min_range, max_range + 1, size=number_points)

np.random.seed(random_seed_c)
random_test_integers_c = np.random.randint(min_range, max_range + 1, size=number_points)

np.random.seed(random_seed_d)
random_test_integers_d = np.random.randint(min_range, max_range + 1, size=number_points)

output = []
i = 1

for i in range(number_points):
    # ground truth
    output.append(random_test_integers_a[i] + random_test_integers_b[i]
                  + random_test_integers_c[i] + random_test_integers_d[i])

d = {'a': random_test_integers_a, 'b': random_test_integers_b,
     'c': random_test_integers_c, 'd': random_test_integers_d, 'y': output}
df_test = pd.DataFrame(d)
print('Data Frame Test Shape : ', df_test.shape)


file_test = open("testing_numbers.csv", "w+")
if file_test == "":
    print('File creation failed')
df_test.to_csv("testing_numbers.csv", ",")

corr = df_test.corr()
print('correlation matrix for testing data : \n', corr)