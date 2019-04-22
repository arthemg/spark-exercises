from __future__ import print_function

from pyspark.ml.regression import LinearRegression

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

if __name__ == "__main__":

    # Creating a SparkSession
    spark = SparkSession.builder.appName("LinearRegression").getOrCreate()

    # Loadind the data and converting it to the MLib standard
    inputLines = spark.sparkContext.textFile("DataSets/regression.txt")
    data = inputLines.map(lambda l: l.split(",")).map(lambda l: (float(l[0]), Vectors.dense(float(l[1]))))

    # Convert RDD to DataFrame
    colNames = ["label", "features"]
    df = data.toDF(colNames)

    # Split the data into training data and testing data
    trainTest = df.randomSplit([0.5, 0.5])
    trainingDF = trainTest[0]
    testDF = trainTest[1]

    # Creating Linear Regression model
    linReg = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # Train the model using our training data
    model = linReg.fit(trainingDF)

    # Check if we can predict values in our test data
    fullPredictions = model.transform(testDF).cache()

    # Exctract the predictions and the "known" correct labels.
    predictions = fullPredictions.select("prediction").rdd.map(lambda l: l[0])
    labels = fullPredictions.select("label").rdd.map(lambda l: l[0])

    # Zip predictions and labels together
    predictionsAndLabels = predictions.zip(labels).collect()

    # Print the predicted and actual values for each point
    for prediction in predictionsAndLabels:
        print(prediction)
