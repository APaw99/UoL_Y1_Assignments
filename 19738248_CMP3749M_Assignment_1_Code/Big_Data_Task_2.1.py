# Task 2 - Section I: Data Summary, Understanding & Visualisation.
#Create the Spark session.
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Create dataframe by reading the 'nuclear_plants_small_dataset.csv' file.
df= spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True)

# Task 2.1: Find any missing values in code, for each column of data.

# Power range sensor 3 column name in dataset has a space key at the end of the column name: 
# "Power_range_sensor_3 "; causes error if the space key is omitted from the variable name in code.

# Rename column to remove extra space key and avoid future errors.
df = df.withColumnRenamed("Power_range_sensor_3 ", "Power_range_sensor_3")

# Pressure sensors column names have space key within the column names:
# Example - "Pressure _sensor_3"; will cause error if space key is omitted from variable name in code.

#Rename pressure sensors to remove extra space key and avoid future errors.
df = df.withColumnRenamed("Pressure _sensor_1", "Pressure_sensor_1") \
        .withColumnRenamed("Pressure _sensor_2", "Pressure_sensor_2") \
        .withColumnRenamed("Pressure _sensor_3", "Pressure_sensor_3") \
        .withColumnRenamed("Pressure _sensor_4", "Pressure_sensor_4")

# Create for loop to print out all column names and display the count of any and all null values within each respective column.
names = df.schema.names
for name in names:
    print(name , 'null value count: ' , df.where(df[name].isNull()).count())

# If there is a null value within a column of data, drop the row of data.
df = df.dropna()

# If there is any duplicate data within the dataframe, drop the duplicate row of data, as it can affect calculations and training sets for machine learning.
df = df.dropDuplicates()

# Show dataframe after renaming column names and removing null data & duplicate data. Show the first 10 rows of the dataframe.
df.show(10)

