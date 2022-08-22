# In this script the PySpark interface for Apache Spark to create a model that predicts wether a patient is at a increased risk of stroke. 

# Libraries
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.sql import SQLContext
from pyspark.sql.functions import mean, col, split, regexp_extract, when, lit
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sklearn.metrics import roc_curve, auc
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initiate Spark
spark = SparkSession.builder.appName('PySpark ML project').getOrCreate()
sc = spark.sparkContext.getOrCreate()

# Import data
# This dataset can be found at Kaggle under 'Brain Stroke Dataset'
path = '../input/brain-stroke-dataset/brain_stroke.csv'
df = spark.read.csv(path, header='True', inferSchema='True')

# Feature engineering
df = df.withColumn(
    'high_risk',
     when((col('age') >= 40) & (col("avg_glucose_level") <= 125), 1).otherwise(0))
  
# Vectorize string data
indexer = StringIndexer(inputCol='gender', outputCol='gender_vec')
df_indexed = indexer.fit(df).transform(df)

indexer2 = StringIndexer(inputCol='ever_married', outputCol='married_vec')
df_indexed2 = indexer2.fit(df_indexed).transform(df_indexed)

indexer3 = StringIndexer(inputCol='work_type', outputCol='work_vec')
df_indexed3 = indexer3.fit(df_indexed2).transform(df_indexed2)

indexer4 = StringIndexer(inputCol='Residence_type', outputCol='residence_vec')
df_indexed4 = indexer4.fit(df_indexed3).transform(df_indexed3)

indexer5 = StringIndexer(inputCol='smoking_status', outputCol='smoking_vec')
df_indexed5 = indexer5.fit(df_indexed4).transform(df_indexed4)

df = df_indexed5.drop(*categorical)
     when((col('age') >= 40) & (col("avg_glucose_level") <= 125), 1).otherwise(0))
  
# Create a feature column
feature = VectorAssembler(inputCols = df.drop('stroke').columns, outputCol='features')
feature_vector = feature.transform(df)

# Select the target and feature columns
# Split the data
ml_df = feature_vector.select(['features', 'stroke'])
train, test = ml_df.randomSplit([0.8, 0.2])

# Logistic Regression model
lr = LogisticRegression(labelCol='stroke')

paramGrid = ParamGridBuilder().addGrid(lr. regParam, (0.01, 0.1))\
                              .addGrid(lr.maxIter, (5, 10))\
                              .addGrid(lr.tol, (1e-4, 1e-5))\
                              .addGrid(lr.elasticNetParam, (0.25, 0.75))\
                              .build()

tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=MulticlassClassificationEvaluator(labelCol='stroke'),
                           trainRatio=0.8)

lr_model = tvs.fit(train)
lr_model_pred = lr_model.transform(test)

# Evaluate LR model
results = lr_model_pred.select(['probability', 'stroke'])

results_collect = results.collect()
results_list = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect]
scoreAndLabels = sc.parallelize(results_list)

metrics = metric(scoreAndLabels)
print('Accuracy:  ', round(MulticlassClassificationEvaluator(labelCol='stroke', metricName='accuracy').evaluate(lr_model_pred), 4))
print('Precision: ', round(MulticlassClassificationEvaluator(labelCol='stroke', metricName='weightedPrecision').evaluate(lr_model_pred), 4))
print('ROC Score: ', round(metrics.areaUnderROC, 4))

# Random Forest model
rf = RandomForestClassifier(labelCol='stroke')

paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 10, 20])\
                              .addGrid(rf.maxBins, [20, 32, 50])\
                              .addGrid(rf.numTrees, [20, 40, 60])\
                              .addGrid(rf.impurity, ['gini', 'entropy'])\
                              .addGrid(rf.minInstancesPerNode, [1, 5, 10])\
                              .build()
    
tvs = TrainValidationSplit(estimator=rf,
                           estimatorParamMaps=paramGrid,
                           evaluator=MulticlassClassificationEvaluator(labelCol='stroke'),
                           trainRatio=0.8)

rf_model = tvs.fit(train)
rf_model_pred = rf_model.transform(test)

# Evaluate RF model
results = rf_model_pred.select(['probability', 'stroke'])

results_collect = results.collect()
results_list = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect]
scoreAndLabels = sc.parallelize(results_list)

metrics = metric(scoreAndLabels)
print('Accuracy:  ', round(MulticlassClassificationEvaluator(labelCol='stroke', metricName='accuracy').evaluate(rf_model_pred), 4))
print('Precision: ', round(MulticlassClassificationEvaluator(labelCol='stroke', metricName='weightedPrecision').evaluate(rf_model_pred), 4))
print('ROC Score: ', round(metrics.areaUnderROC, 4))

# Thanks for reading!
