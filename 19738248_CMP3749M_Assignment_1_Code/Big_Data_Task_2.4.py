# Task 2 - Section II: Classification & Big Data Analysis.
# Create the Spark session.
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Import the libraries that are needed in this assignment.
import pandas as pd
import numpy as np

df2 = pd.read_csv("nuclear_plants_small_dataset.csv")
df2 = df2.rename(columns={'Power_range_sensor_3 ': 'Power_range_sensor_3', 'Pressure_sensor_1': 'Pressure_sensor_1', 'Pressure _sensor_2': 'Pressure_sensor_2',
                          'Pressure _sensor_3': 'Pressure_sensor_3', 'Pressure _sensor_4': 'Pressure_sensor_4'})

# Task 4 - Shuffle the dataset & split dataset into a 70/30 ratio - 70% Training set, 30% test set.
df2_rand = df2.sample(frac=1)
display(df2_rand)

# Create the training set of data from the dataframe. The training set is randomised.
df2_training_set = df2_rand.sample(frac = 0.7)
display(df2_training_set)

# Create the test set of data from the dataframe. The test set is randomised. Contains the remainder data excluded from the training set of data.
df2_test_set = df2.drop(df2_training_set.index)
display(df2_test_set)
