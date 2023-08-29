# Task 2 - Section II: Classification & Big Data Analysis.
# Create the Spark session.
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Task 5 - Setup training and test sets of data; train a decision tree, support vector machine model & artificial neural network with training set & apply classifiers to test set.
# Task 5, Part 1: Create the Decision Tree Classifier.
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import math

df = spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True)

df = df.withColumnRenamed("Power_range_sensor_3 ", "Power_range_sensor_3") \
.withColumnRenamed("Pressure _sensor_1", "Pressure_sensor_1") \
.withColumnRenamed("Pressure _sensor_2", "Pressure_sensor_2") \
.withColumnRenamed("Pressure _sensor_3", "Pressure_sensor_3") \
.withColumnRenamed("Pressure _sensor_4", "Pressure_sensor_4")

# Setup the vector assembler to create a column called 'Features' that combines all columns together.
vector_assembler = VectorAssembler(inputCols= ["Power_range_sensor_1", "Power_range_sensor_2", "Power_range_sensor_3", "Power_range_sensor_4",
                                               "Pressure_sensor_1", "Pressure_sensor_2", "Pressure_sensor_3", "Pressure_sensor_4",
                                               "Vibration_sensor_1", "Vibration_sensor_2", "Vibration_sensor_3", "Vibration_sensor_4"],
                                  outputCol="features")

df_temp = vector_assembler.transform(df)

# Drop the individuals columns and retain the 'features' columns.
df_dt = df_temp.drop("Power_range_sensor_1", "Power_range_sensor_2", "Power_range_sensor_3", "Power_range_sensor_4",
                                               "Pressure_sensor_1", "Pressure_sensor_2", "Pressure_sensor_3", "Pressure_sensor_4",
                                               "Vibration_sensor_1", "Vibration_sensor_2", "Vibration_sensor_3", "Vibration_sensor_4")


# Index the 'Status' column. StatusIndex renames row values: "Normal" and "Abnormal" as "0" and "1"
Status_Indexer = StringIndexer(inputCol="Status", outputCol="StatusIndex").fit(df_dt)

# Index the columns.
feature_Indexer = VectorIndexer(inputCol="features", outputCol="FeaturesIndex", maxCategories=2).fit(df_dt)

# Split the dataframe into two sets - training & test set.
(Training_df, Test_df) = df_dt.randomSplit([0.7, 0.3])

# Train a Decision Tree model.
Decision_Tree = DecisionTreeClassifier(labelCol="StatusIndex", featuresCol="features")

# Create a pipeline to chain the decision tree and indexers.
pipeline = Pipeline(stages=[Status_Indexer, feature_Indexer, Decision_Tree])

# Train the decision tree model & run the indexers.
model = pipeline.fit(Training_df)

# Create the predictions variable.
Predictions = model.transform(Test_df)

# Display the example rows.
Predictions.select("prediction", "StatusIndex", "features").show(10)

# Compute the test error of the decision tree classifier method.
evaluator = MulticlassClassificationEvaluator(labelCol="StatusIndex",
                                              predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(Predictions)
print("Test Error = %g " % (1.0 - accuracy))

treeModel = model.stages[2]
# Show summary.
print(treeModel)
df_dt.show(5)

# Create variables for: true positive, true negative, false positive, false negative.
# "0" is treated as positive and "1" as negative.
tp = Predictions.filter((Predictions.prediction == 0) & (Predictions.StatusIndex == 0))
tpval = tp.count()
print("True Positive Values = ", tpval)
tn = Predictions.filter((Predictions.prediction == 1) & (Predictions.StatusIndex == 1))
tnval = tn.count()
print("True Negative Values = ", tnval)
fp = Predictions.filter((Predictions.prediction == 0) & (Predictions.StatusIndex == 1))
fpval = fp.count()
print("False Positive Values = ", fpval)
fn = Predictions.filter((Predictions.prediction == 1) & (Predictions.StatusIndex == 0))
fnval = fn.count()
print("False Negative Values = ", fnval)

# Calculate specificity and sensitivity.
# Sensitivity = True Positives / (True Positives + False Negatives)
Sensitivity = tpval/(tpval + fnval)
print("Decision Tree Sensitivity = ", Sensitivity)
# Specificity = True Negative / (True Negatives + False Positives)
Specificity = tnval/(tnval + fpval)
print("Decision Tree Specificity = ", Specificity)
Error_rate = 1-accuracy
print("Decision Tree Error rate: ", Error_rate)

# Task 5, Part 2: Create the Support Vector Machine Model.
from pyspark.ml.classification import LinearSVC

df_VCM_train = df_dt

# Rename 'Status' column into 'label.
String_Indexer_training = StringIndexer(inputCol='Status', outputCol='label').fit(df_VCM_train)
df_VCM_train = String_Indexer_training.transform(df_VCM_train)

lsvc = LinearSVC(maxIter=10, regParam=0.1)

# Fit the data into the model.
lvscModel = lsvc.fit(df_VCM_train)

# Print the coefficents and intercept for the linear SVC.

print("Coefficients: " + str(lvscModel.coefficients))
print("Intercept: " + str(lvscModel.intercept))

# Compute the test accuracy on the test set.
result = lvscModel.transform(df_VCM_train)
predictionVCM = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionVCM)))

# Create variables for: true positive, true negative, false positive, false negative.
# "0" is treated as positive and "1" as negative.
tp = predictionVCM.filter((predictionVCM.prediction == 0) & (predictionVCM.label == 0))
tpval = tp.count()
print("True Positive Values = ", tpval)
tn = predictionVCM.filter((predictionVCM.prediction == 1) & (predictionVCM.label == 1))
tnval = tn.count()
print("True Negative Values = ", tnval)
fp = predictionVCM.filter((predictionVCM.prediction == 0) & (predictionVCM.label == 1))
fpval = fp.count()
print("False Positive Values = ", fpval)
fn = predictionVCM.filter((predictionVCM.prediction == 1) & (predictionVCM.label == 0))
fnval = fn.count()
print("False Negative Values = ", fnval)

# Calculate specificity and sensitivity.
# Sensitivity = True Positives / (True Positives + False Negatives)
Sensitivity = tpval/(tpval + fnval)
print("Vector Machine Sensitivity = ", Sensitivity)
# Specificity = True Negative / (True Negatives + False Positives)
Specificity = tnval/(tnval + fpval)
print("Vector Machine Specificity = ", Specificity)
Error_rate = 1-evaluator.evaluate(predictionVCM)
print("Vector Machine Error rate: ", Error_rate)

# Task 5, Part 3: Create an Artificial Neural Network.

# Copy the 'df_dt' variable, split the data into 70/30 ratio, training set and test set respectively.
df_ANN = df_dt

# Rename 'Status' column into 'label.
String_Indexer_ANN = StringIndexer(inputCol='Status', outputCol='label').fit(df_ANN)
df_ANN = String_Indexer_training.transform(df_ANN)

# Split the dataframe into two sets of data, 'train' and 'test'.
splits = df_ANN.randomSplit([0.7, 0.3])
train_ANN = splits[0]
test_ANN = splits[1]

# Create layers for the neural network: Input Layer 1 & Output Layer 2.
layers = [12, 2]

# Create the trainer and setup its parameters.
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

# Train the model.
model = trainer.fit(train_ANN)

# Compute accuracy on the test set.
result = model.transform(test_ANN)
predictionANN = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionANN)))

# Calculate sensitivity and Specificity.
# Create variables for: true positive, true negative, false positive, false negative.
# "0" is treated as positive and "1" as negative.
tp = predictionANN.filter((predictionANN.prediction == 0) & (predictionANN.label == 0))
tpval = tp.count()
print("True Positive Values = ", tpval)
tn = predictionANN.filter((predictionANN.prediction == 1) & (predictionANN.label == 1))
tnval = tn.count()
print("True Negative Values = ", tnval)
fp = predictionANN.filter((predictionANN.prediction == 0) & (predictionANN.label == 1))
fpval = fp.count()
print("False Positive Values = ", fpval)
fn = predictionANN.filter((predictionANN.prediction == 1) & (predictionANN.label == 0))
fnval = fn.count()
print("False Negative Values = ", fnval)

# Calculate specificity and sensitivity.
# Sensitivity = True Positives / (True Positives + False Negatives)
Sensitivity = tpval/(tpval + fnval)
print("ANN Sensitivity = ", Sensitivity)
# Specificity = True Negative / (True Negatives + False Positives)
Specificity = tnval/(tnval + fpval)
print("ANN Specificity = ", Specificity)
Error_rate = 1-evaluator.evaluate(predictionANN)
print("ANN Error rate: ", Error_rate)
