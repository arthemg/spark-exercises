from pyspark import SparkConf, SparkContext
import re

conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf=conf)


def normalizeWords(text):
    return re.compile(r'\W+', re.UNICODE).split(text.lower())


input = sc.textFile("DataSets/Book.txt")
words = input.flatMap(normalizeWords)
wordCounts = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)


wordSorted = wordCounts.map(lambda x: (x[1], x[0])).sortByKey()
results = wordSorted.collect()

for result in results:
    count = str(result[0])
    word = result[1].encode('ascii', 'ignore')
    if(word):
        print(word.decode() + ":\t\t\t" + count)