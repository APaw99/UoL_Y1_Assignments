# Task 2 - Section II: Classification & Big Data Analysis.
# Create the Spark session.
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

#Task 8 - Use mapReduce to collect summary statistics of the dataset.
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.rdd import RDD
import math
import numpy as np
#Read the CSV file. Rename all columns to make sure no errors or typos exist each column name.
big_df = spark.read.csv("nuclear_plants_big_dataset.csv",inferSchema=True,header=True)

big_df = big_df.withColumnRenamed("Power_range_sensor_3 ", "Power_range_sensor_3") \
.withColumnRenamed("Pressure _sensor_1", "Pressure_sensor_1") \
.withColumnRenamed("Pressure _sensor_2", "Pressure_sensor_2") \
.withColumnRenamed("Pressure _sensor_3", "Pressure_sensor_3") \
.withColumnRenamed("Pressure _sensor_4", "Pressure_sensor_4")

big_vector_assembler = VectorAssembler(inputCols= ["Power_range_sensor_1", "Power_range_sensor_2", "Power_range_sensor_3",
                                                   "Power_range_sensor_4", "Pressure_sensor_1", "Pressure_sensor_2", "Pressure_sensor_3",
                                                   "Pressure_sensor_4", "Vibration_sensor_1", "Vibration_sensor_2", "Vibration_sensor_3",
                                                   "Vibration_sensor_4"], outputCol="Sensors")
big_df_vector = big_vector_assembler.transform(big_df)

big_df_vector = big_df_vector.drop("Power_range_sensor_1", "Power_range_sensor_2", "Power_range_sensor_3",
                     "Power_range_sensor_4", "Pressure_sensor_1", "Pressure_sensor_2", "Pressure_sensor_3",
                     "Pressure_sensor_4", "Vibration_sensor_1", "Vibration_sensor_2", "Vibration_sensor_3",
                     "Vibration_sensor_4")

big_df = big_df.drop("Status")
#Calculate the summary statistics for each column: Min, Max & Mean.
df_rdd = big_df.rdd
df_rdd_vec = big_df_vector.rdd
df_rdd_avg = df_rdd_vec.mapValues(lambda v: (v, 1)) \
.reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])) \
.mapValues(lambda v: v[0]/v[1]) \
.collectAsMap()

print("Average/Mean of each sensor column: ", df_rdd_avg)

df_max = df_rdd.max()
df_min = df_rdd.min()

print("Max value of each sensor column: ", df_max)
print("Min value of each sensor column: ", df_min)
