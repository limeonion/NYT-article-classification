from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import *

from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName("NeuralNetworks")
conf = conf.setMaster("local[*]")
sc   = SparkContext(conf=conf)

sqlContext = SQLContext(sc)

# Load and parse the data file, converting it to a DataFrame.
trainingData = sqlContext.read.format("libsvm").option("delimiter", " ").load("../../data/featureMatrixTrainingdata.txt")
testData = sqlContext.read.format("libsvm").option("delimiter", " ").load("../../data/featureMatrixTestingdata.txt")
# Load and parse the data file, converting it to a DataFrame.

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(trainingData)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(trainingData)

# Split the data into training and test sets (30% held out for testing)
#(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)
predictions.show()
# Select example rows to display.
#predictions.select("predictedLabel", "label", "features").show(60, False)
predictionAndLabels = predictions.select(col("prediction"), col("indexedLabel"))
#predictionAndLabels.show(60, False)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))
print("Accuracy " + str(accuracy))

#Print the confusion matrix of prediction on test data
metrics = MulticlassMetrics(predictionAndLabels.rdd)
print("Confusion Matrix:\n" + str(metrics.confusionMatrix().toArray()))

