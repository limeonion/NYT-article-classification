from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel

from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

conf = SparkConf().setAppName("NeuralNetworks")
conf = conf.setMaster("local[*]")
sc   = SparkContext(conf=conf)

sqlContext = SQLContext(sc)

# Load training data

df = sqlContext.read.format("libsvm").option("delimiter", " ").option("header", "false").load("test_data.txt")

#print("Dataframe")
df.show()
(trainingData, testData) = df.randomSplit([0.7, 0.3])

# Load training data
#training = sqlContext \
#    .read \
#    .format("libsvm") \
#    .load("doc2.txt")

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

print("*****************************Training Data**********************")
#trainingData.show()
print("******************************TestData *************************")
#testData.show()
# Fit the model
lrModel = lr.fit(trainingData)

# Print the coefficients and intercept for multinomial logistic regression
#print("Coefficients: \n" + str(lrModel.coefficientMatrix))
#print("Intercept: " + str(lrModel.interceptVector))


# We can also use the multinomial family for binary classification
mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")

# Fit the model
mlrModel = mlr.fit(trainingData)

# Print the coefficients and intercepts for logistic regression with multinomial family
#print("Multinomial coefficients: " + str(mlrModel.coefficientMatrix))
#print("Multinomial intercepts: " + str(mlrModel.interceptVector))
    # $example off$

test_data_label = lrModel.transform(testData)

evaluator = MulticlassClassificationEvaluator()
accuracy = evaluator.evaluate(test_data_label)
#test_data_label.select("prediction", "label").show(60, False)

#evaluator = BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")

#accuracy = evaluator.evaluate(test_data_label)

print("Accuracy: " + str(accuracy))



# Reference : https://www.codementor.io/jadianes/spark-mllib-logistic-regression-du107neto

#mlrm = LogisticRegressionModel(mlrModel.coefficientMatrix, mlrModel.interceptVector, 70, 4)

#test_label =mlrm.evaluate(testData)

#test_label.show()


#labels_and_preds = testData.rdd.map(lambda p: (p.label, mlrModel.predict(p.features)))

#t0 = time()
#test_accuracy = labels_and_preds.filter(lambda (v, p): v == p).count() / float(60)
#tt = time() - t0

#print ("Prediction made in {} seconds. Test accuracy is {}", round(test_accuracy,4))