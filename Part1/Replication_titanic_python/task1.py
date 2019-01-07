# Import packages
import time
import pyspark
import os
import csv
from numpy import array
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf

# Creating Spark environment
os.environ["HADOOP_USER_NAME"] = "hdfs"
os.environ["PYTHON_VERSION"] = "3.5.2"
conf = pyspark.SparkConf()
sc = pyspark.SparkContext(conf=conf)
conf.getAll()

# Reading from the hdfs, removing the header
trainTitanic = sc.textFile("/home/hadoop/Desktop/train.csv")
trainHeader = trainTitanic.first()
trainTitanic = trainTitanic.filter(lambda line: line != trainHeader).mapPartitions(lambda x: csv.reader(x))
trainTitanic.first()
 
# Data preprocessing
def sexTransformMapper(elem):
    '''Function which transform "male" into 1 and else things into 0
    - elem : string
    - return : vector
    '''
     
    if elem == 'male' :
        return [0]
    else :
        return [1]
 
# Data Transformations and filter lines with empty strings
trainTitanic=trainTitanic.map(lambda line: line[1:3]+sexTransformMapper(line[4])+line[5:11])
trainTitanic=trainTitanic.filter(lambda line: line[3] != '' ).filter(lambda line: line[4] != '' )
trainTitanic.take(10)
 
# creating "labeled point" rdds specific to MLlib "(label (v1, v2...vp])"
trainTitanicLP=trainTitanic.map(lambda line: LabeledPoint(line[0],[line[1:5]]))
trainTitanicLP.first()
 
# splitting dataset into train and test set
(trainData, testData) = trainTitanicLP.randomSplit([0.7, 0.3])
 
# Random forest : same parameters as sklearn (?)
from pyspark.mllib.tree import RandomForest
 
time_start=time.time()
model_rf = RandomForest.trainClassifier(trainData, numClasses = 2,
        categoricalFeaturesInfo = {}, numTrees = 100,
        featureSubsetStrategy='auto', impurity='gini', maxDepth=12,
        maxBins=32, seed=None)
 
  
model_rf.numTrees()
model_rf.totalNumNodes()
time_end=time.time()
time_rf=(time_end - time_start)
print("RF takes %d s" %(time_rf))
 
# Predictions on test set
predictions = model_rf.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
 
# first metrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics
metrics = BinaryClassificationMetrics(labelsAndPredictions)
 
# Area under precision-recall curve
print("Area under PR = %s" % metrics.areaUnderPR)
 
# Area under ROC curve
print("Area under ROC = %s" % metrics.areaUnderROC)

# Import packages
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler, IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import *
 
# Creatingt Spark SQL environment
from pyspark.sql import SparkSession, HiveContext
SparkContext.setSystemProperty("hive.metastore.uris", "thrift://nn1:9083")
spark = SparkSession.builder.enableHiveSupport().getOrCreate()
 
# spark is an existing SparkSession
train = spark.read.csv("/home/hadoop/Desktop/train.csv", header = True)
# Displays the content of the DataFrame to stdout
train.show(10)
 
# String to float on some columns of the dataset : creates a new dataset
train = train.select(col("Survived"),col("Sex"),col("Embarked"),col("Pclass").cast("float"),col("Age").cast("float"),col("SibSp").cast("float"),col("Fare").cast("float"))
 
# dropping null values
train = train.dropna()
 
# Spliting in train and test set. Beware : It sorts the dataset
(traindf, testdf) = train.randomSplit([0.7,0.3])


# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
train = StringIndexer(inputCol="Sex", outputCol="indexedSex").fit(train).transform(train)
train = StringIndexer(inputCol="Embarked", outputCol="indexedEmbarked").fit(train).transform(train)
 
train = StringIndexer(inputCol="Survived", outputCol="indexedSurvived").fit(train).transform(train)
 
# One Hot Encoder on indexed features
train = OneHotEncoder(inputCol="indexedSex", outputCol="sexVec").transform(train)
train = OneHotEncoder(inputCol="indexedEmbarked", outputCol="embarkedVec").transform(train)
 
# Feature assembler as a vector
train = VectorAssembler(inputCols=["Pclass","sexVec","embarkedVec", "Age","SibSp","Fare"],outputCol="features").transform(train)
 
rf = RandomForestClassifier(labelCol="indexedSurvived", featuresCol="features")
 
model = rf.fit(train)
 
predictions = model.transform(train)
 
# Select example rows to display.
predictions.select(col("prediction"),col("probability"),).show(5)

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
genderIndexer = StringIndexer(inputCol="Sex", outputCol="indexedSex")
embarkIndexer = StringIndexer(inputCol="Embarked", outputCol="indexedEmbarked")
 
surviveIndexer = StringIndexer(inputCol="Survived", outputCol="indexedSurvived")
 
# One Hot Encoder on indexed features
genderEncoder = OneHotEncoder(inputCol="indexedSex", outputCol="sexVec")
embarkEncoder = OneHotEncoder(inputCol="indexedEmbarked", outputCol="embarkedVec")
 
# Create the vector structured data (label,features(vector))
assembler = VectorAssembler(inputCols=["Pclass","sexVec","Age","SibSp","Fare","embarkedVec"],outputCol="features")
 
# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedSurvived", featuresCol="features")
 
# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[surviveIndexer, genderIndexer, embarkIndexer, genderEncoder,embarkEncoder, assembler, rf]) # genderIndexer,embarkIndexer,genderEncoder,embarkEncoder,
 
# Train model.  This also runs the indexers.
model = pipeline.fit(traindf)
 
# Predictions
predictions = model.transform(testdf)
 
# Select example rows to display.
predictions.columns 
 
# Select example rows to display.
predictions.select("prediction", "Survived", "features").show(5)
 
# Select (prediction, true label) and compute test error
predictions = predictions.select(col("Survived").cast("Float"),col("prediction"))
evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))
 
rfModel = model.stages[6]
print(rfModel)  # summary only
 
evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % accuracy)
 
evaluatorf1 = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1")
f1 = evaluatorf1.evaluate(predictions)
print("f1 = %g" % f1)
 
evaluatorwp = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedPrecision")
wp = evaluatorwp.evaluate(predictions)
print("weightedPrecision = %g" % wp)
 
evaluatorwr = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedRecall")
wr = evaluatorwr.evaluate(predictions)
print("weightedRecall = %g" % wr)
 
# close sparkcontext
sc.stop()


