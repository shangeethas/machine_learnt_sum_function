# libraries
import numpy as np
import pandas as pd

print("Welcome to machine learnt Sum Function")

print("We're generating training data set")
random_seed_text = input("Enter random seed : ")
number_points_text = input("Enter number of data points required : ")
range_min_text = input("Enter minimum range of data points : ")
range_max_text = input("Enter maximum range of data points : ")

random_seed = int(random_seed_text)
number_points = int(number_points_text)
range_min = int(range_min_text)
range_max = int(range_max_text)

if range_min == range_max:
    print('Minimum and Maximum values of Range should be different')
    quit()
elif range_min > range_max:
    print('Range Minimum should be less than Range Maximum value')
    quit()
else:
    print('Valid inputs are provided by the user')

np.random.seed(random_seed)
random_train_integers = np.random.randint(range_min, range_max + 1, size=number_points)

random_train_integers = random_train_integers.reshape(-1, 1)
df_train = pd.DataFrame.from_records(random_train_integers)

file_train = open("training_numbers.csv", "w+")
if file_train == "":
    print('File creation failed')
    quit()


df_train.to_csv("training_numbers.csv", ",")

print("We're generating test data set")
random_seed_text = input("Enter random seed : ")
number_points_text = input("Enter number of data points required : ")
range_min_text = input("Enter minimum range of data points : ")
range_max_text = input("Enter maximum range of data points : ")

random_seed = int(random_seed_text)
number_points = int(number_points_text)
range_min = int(range_min_text)
range_max = int(range_max_text)

if range_min == range_max:
    print('Minimum and Maximum values of Range should be different')
    quit()
elif range_min > range_max:
    print('Range Minimum should be less than Range Maximum value')
    quit()
else:
    print('Valid inputs are provided by the user')

np.random.seed(random_seed)
random_test_integers = np.random.randint(range_min, range_max + 1, size=number_points)

random_test_integers = random_test_integers.reshape(-1, 1)
df_test = pd.DataFrame.from_records(random_test_integers)

file_test = open("testing_numbers.csv", "w+")
if file_test == "":
    print('File creation failed')
    quit()


df_test.to_csv("testing_numbers.csv", ",")
