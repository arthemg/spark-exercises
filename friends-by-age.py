from pyspark import SparkConf, SparkContext
import collections

conf = SparkConf().setMaster("local").setAppName("Friends-By-Age")
sc = SparkContext(conf=conf)


def parseLine(line):
    fields = line.split(',')
    age = int(fields[2])
    numFriends = int(fields[3])
    return age, numFriends


lines = sc.textFile("DataSets/fakefriends.csv")
rdd = lines.map(parseLine)
totalByAge = rdd.mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
averageByAge = totalByAge.mapValues(lambda x: x[0] / x[1])
results = averageByAge.collect()
for result in results:
    print(result)
