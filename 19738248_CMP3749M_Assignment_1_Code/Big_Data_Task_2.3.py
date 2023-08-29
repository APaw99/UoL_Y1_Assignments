# Task 2 - Section I: Data Summary, Understanding & Visualisation.
# Create the Spark session.
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Create dataframe by reading the 'nuclear_plants_small_dataset.csv' file.
df2 = pd.read_csv("nuclear_plants_small_dataset.csv")

# Import libraries that are dependencies for this task.
import pandas as pd
import matplotlib as plt
import numpy as np

# Task 2.3 - Create and display correlation matrices for the features of the dataframe.

# Create a copy of the dataframe in pandas to use for correlation matrix.
df2_copy_1 = df2.copy()

df2_copy_1.columns = ["Status", "Power_range_sensor_1", "Power_range_sensor_2", "Power_range_sensor_3", "Power_range_sensor_4",
                                                   "Pressure_sensor_1", "Pressure_sensor_2", "Pressure_sensor_3", "Pressure_sensor_4", 
                                                   "Vibration_sensor_1", "Vibration_sensor_2", "Vibration_sensor_3", "Vibration_sensor_4"]
# Set the column 'Status' as the index for the dataframe.
df2_copy_1 = df2_copy_1.set_index("Status")
# Drop 'Abnormal' status from the dataframe copy to only contain 'Normal' Status values.
df2_copy_1 = df2_copy_1.drop("Abnormal")

# Drop the 'Status' column as it does not contain any integer or float values.
df2_copy_1 = df2_copy_1.reset_index(drop=True)

# Create the matrix variable to setup the data into a correlation matrix.
matrix_1 = df2_copy_1.corr()
# Display the correlation matrix for 'Normal' Status values.
display(matrix_1)

# Create a second copy of the dataframe to use for the second correlation matrix.
df2_copy_2 = df2.copy()

df2_copy_2.columns = ["Status", "Power_range_sensor_1", "Power_range_sensor_2", "Power_range_sensor_3", "Power_range_sensor_4",
                                                   "Pressure_sensor_1", "Pressure_sensor_2", "Pressure_sensor_3", "Pressure_sensor_4", 
                                                   "Vibration_sensor_1", "Vibration_sensor_2", "Vibration_sensor_3", "Vibration_sensor_4"]


# Drop all 'Normal' values from the index to have the dummy dataframe contain only 'Abnormal' status values.
df2_copy_2 = df2_copy_2.set_index("Status")
df2_copy_2 = df2_copy_2.drop("Normal")

# Drop the index column 'Status' in the 2nd copy of the dataframe.
df2_copy_2 = df2_copy_2.reset_index(drop=True)

# Display the correlation matrix for the 'abnormal' status values.
matrix_2 = df2_copy_2.corr()
display(matrix_2)
