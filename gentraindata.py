# libraries
import numpy as np
import pandas as pd
import yaml as yaml

with open("configs_train.yaml", 'r') as stream:
    try:
        configs = yaml.safe_load(stream)
        print(configs)
    except yaml.YAMLError as exc:
        print(exc)

random_seed_a = configs.get('random_seed_a')
random_seed_b = configs.get('random_seed_b')
random_seed_c = configs.get('random_seed_c')
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
random_train_integers_a = np.random.randint(min_range, max_range + 1, size=number_points)

np.random.seed(random_seed_b)
random_train_integers_b = np.random.randint(min_range, max_range + 1, size=number_points)

np.random.seed(random_seed_c)
random_train_integers_c = np.random.randint(min_range, max_range + 1, size=number_points)

output = []
i = 1

for i in range(number_points):
    # ground truth
    output.append(random_train_integers_a[i] + random_train_integers_b[i] + random_train_integers_c[i])

d = {'a': random_train_integers_a, 'b': random_train_integers_b, 'c': random_train_integers_c, 'y': output}
df_train = pd.DataFrame(d)
print('Data Frame Train Shape : ', df_train.shape)


file_train = open("training_numbers.csv", "w+")
if file_train == "":
    print('File creation failed')
df_train.to_csv("training_numbers.csv", ",")

corr = df_train.corr()
print('correlation matrix for training data : \n', corr)

