from pyspark import SparkConf, SparkContext
import collections

conf = SparkConf().setMaster("local").setAppName("MinTemperatures")
sc = SparkContext(conf=conf)


def parseline(line):
    fields = line.split(',')
    stationID = fields[0]
    entryType = fields[2]
    temperature = float(fields[3]) * 0.1 * (9.0 / 5.0) + 32.0
    return stationID, entryType, temperature


lines = sc.textFile("DataSets/1800.csv")
parsedLines = lines.map(parseline)
minTemps = parsedLines.filter(lambda x: "TMIN" in x[1])
stationsTemps = minTemps.map(lambda x: (x[0], x[2]))
minTemps = stationsTemps.reduceByKey(lambda x, y: min(x, y))
results = minTemps.collect()
for result in results:
    print(result[0] + "\t{:.2f}F".format(result[1]))
