# Task 2 - Section I: Data Summary, Understanding & Visualisation.

# Create the Spark session.
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Task 2.2 - Show summary statistics for columns grouped by 'Normal' and 'Abnormal' status from 'Status' column.

# Create dataframe by reading the 'nuclear_plants_small_dataset.csv' file.
df= spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True)

# Import all libraries that are dependencies.
import pandas as pd
import numpy as np

# Display summary table for count, min, max, mean & median statistics.
df1 = df.summary("count", "min", "max", "mean", "50%")
df1.show()

# Show statistics for unbiased variance for every column, grouped by 'Status'.

df2 = pd.read_csv("nuclear_plants_small_dataset.csv")
df2 = df2.rename(columns={'Power_range_sensor_3 ': 'Power_range_sensor_3', 'Pressure_sensor_1': 'Pressure_sensor_1', 'Pressure _sensor_2': 'Pressure_sensor_2',
                          'Pressure _sensor_3': 'Pressure_sensor_3', 'Pressure _sensor_4': 'Pressure_sensor_4'})
df3 = df2.groupby(['Status']).var()
display(df3)

# Show statistics for the mode of each column.
# Each column may contain more than one mode value.
df4 = df2.mode(axis=0, dropna=True)
display(df4)
display(df2.groupby(['Status']).describe())

#Task 2 Continued - setting up and displaying box plots of each 'feature' of the dataset - the columns and the values stored in the dataset rows, grouped by 'Normal' and 'Abnormal' status.

#Create a dummy dataset that is a deep copy of 'df2' - the dataset read in Pandas.
df2_dummy_1 = df2.copy()

df2_dummy_1.columns = ["Status", "Power_range_sensor_1", "Power_range_sensor_2", "Power_range_sensor_3", "Power_range_sensor_4",
                                                   "Pressure_sensor_1", "Pressure_sensor_2", "Pressure_sensor_3", "Pressure_sensor_4", 
                                                   "Vibration_sensor_1", "Vibration_sensor_2", "Vibration_sensor_3", "Vibration_sensor_4"]

# Set the column 'Status' as the index for the dataframe.
df2_dummy_1 = df2_dummy_1.set_index("Status")
# Drop all 'Normal' values from the index to have the dummy dataframe contain only 'Abnormal' status values.
df2_dummy_1 = df2_dummy_1.drop("Normal")
display(df2_dummy_1)

# Create a second dummy dataset that is a deep copy of 'df2'.
df2_dummy_2 = df2.copy()

df2_dummy_2.columns = ["Status", "Power_range_sensor_1", "Power_range_sensor_2", "Power_range_sensor_3", "Power_range_sensor_4",
                                                   "Pressure_sensor_1", "Pressure_sensor_2", "Pressure_sensor_3", "Pressure_sensor_4", 
                                                   "Vibration_sensor_1", "Vibration_sensor_2", "Vibration_sensor_3", "Vibration_sensor_4"]

df2_dummy_2 = df2_dummy_2.set_index("Status")
# Drop all 'Abnormal' values from the index to have the dummy dataframe contain only 'Normal' status values.
df2_dummy_2 = df2_dummy_2.drop("Abnormal")
display(df2_dummy_2)

# Setup variables for matplotlib, for 'df2_dummy_1' dataframe.
Power_range_sensor_1 = df2_dummy_1["Power_range_sensor_1"]
Power_range_sensor_2 = df2_dummy_1["Power_range_sensor_2"]
Power_range_sensor_3 = df2_dummy_1["Power_range_sensor_3"]
Power_range_sensor_4 = df2_dummy_1["Power_range_sensor_4"]
Pressure_sensor_1 = df2_dummy_1["Pressure_sensor_1"]
Pressure_sensor_2 = df2_dummy_1["Pressure_sensor_2"]
Pressure_sensor_3 = df2_dummy_1["Pressure_sensor_3"]
Pressure_sensor_4 = df2_dummy_1["Pressure_sensor_4"]
Vibration_sensor_1 = df2_dummy_1["Vibration_sensor_1"]
Vibration_sensor_2 = df2_dummy_1["Vibration_sensor_2"]
Vibration_sensor_3 = df2_dummy_1["Vibration_sensor_3"]
Vibration_sensor_4 = df2_dummy_1["Vibration_sensor_4"]

# Setup variables for matplotlib, for the 'df2_dummy_2' dataframe. Variables have been abbreviated to distinguish them from variables for the 'df2_dummy_1' dataframe.
# Example: Power_range_sensor_1 is PRS1.
PRS1 = df2_dummy_2["Power_range_sensor_1"]
PRS2 = df2_dummy_2["Power_range_sensor_2"]
PRS3 = df2_dummy_2["Power_range_sensor_3"]
PRS4 = df2_dummy_2["Power_range_sensor_4"]
PS1 = df2_dummy_2["Pressure_sensor_1"]
PS2 = df2_dummy_2["Pressure_sensor_2"]
PS3 = df2_dummy_2["Pressure_sensor_3"]
PS4 = df2_dummy_2["Pressure_sensor_4"]
VS1 = df2_dummy_2["Vibration_sensor_1"]
VS2 = df2_dummy_2["Vibration_sensor_2"]
VS3 = df2_dummy_2["Vibration_sensor_3"]
VS4 = df2_dummy_2["Vibration_sensor_4"]


# Create columns variable for all variables that pull a column from the 'df2_dummy_1' dataframe.
columns_dummy_1 = [Power_range_sensor_1, Power_range_sensor_2, Power_range_sensor_3, Power_range_sensor_4,
                                                   Pressure_sensor_1, Pressure_sensor_2, Pressure_sensor_3, Pressure_sensor_4, 
                                                   Vibration_sensor_1, Vibration_sensor_2, Vibration_sensor_3, Vibration_sensor_4]

columns_dummy_2 = [PRS1, PRS2, PRS3, PRS4, PS1, PS2, PS3, PS4, VS1, VS2, VS3, VS4]


# Plot a box plot for the values across all columns within the 'df2_dummy_1' dataframe.
fig1, ax1 = plt.subplots()
ax1.boxplot(columns_dummy_1, meanline=True, showmeans=True)

plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ["PRS1", "PRS2", "PRS3", "PRS4",
                                                     "PS1", "PS2", "PS3", "PS4",
                                                    "VS1", "VS2", "VS3", "VS4"], rotation=5)

# Plot a box plot for all the values across all columns within the 'df2_dummy_2' dataframe.
Fig2, ax2 = plt.subplots()
ax2.boxplot(columns_dummy_2, meanline=True, showmeans=True)

plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ["PRS1", "PRS2", "PRS3", "PRS4",
                                                     "PS1", "PS2", "PS3", "PS4",
                                                    "VS1", "VS2", "VS3", "VS4"], rotation=5)

plt.show()
