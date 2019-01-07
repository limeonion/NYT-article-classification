# -*- coding: utf-8 -*-
"""
Created on Tue May  8 01:32:51 2018

@author: hadoop
"""

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

# Load training data
from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName("NeuralNetworks")
conf = conf.setMaster("local[*]")
sc   = SparkContext(conf=conf)

sqlContext = SQLContext(sc)

# Load and parse the data file, converting it to a DataFrame.
#data = sqlContext.read.format("libsvm").option("delimiter", " ").load("test_data.txt")
# Load and parse the data file, converting it to a DataFrame.
#data = sqlContext.read.format("libsvm").option("delimiter", " ").load("doc2.txt")
train = sqlContext.read.format("libsvm").option("delimiter", " ").load("../../data/featureMatrixTrainingdata.txt")
test = sqlContext.read.format("libsvm").option("delimiter", " ").load("../../data/featureMatrixTestingdata.txt")


# Split the data into train and test
#splits = data.randomSplit([0.6, 0.4], 1234)
#train = splits[0]
#test = splits[1]

# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# train the model
model = nb.fit(train)

# select example rows to display.
predictions = model.transform(test)
predictions.show()
predictionAndLabels = predictions.select("prediction", "label")
# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))


#Print the confusion matrix of prediction on test data
metrics = MulticlassMetrics(predictionAndLabels.rdd)
#print(metrics.confusionMatrix().toArray())
print("Confusion Matrix:\n" + str(metrics.confusionMatrix().toArray()))