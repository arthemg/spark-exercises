from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("MostPopularSuperhero")
sc = SparkContext(conf=conf)

def countCoOccurences(line):
    elements = line.split()
    return (int(elements[0]), len(elements) - 1)


def parseName(line):
    fields = line.split('\"')
    return (int(fields[0]), fields[1].encode("utf8"))


names = sc.textFile("DataSets/Marvel-Data/Marvel-Names.txt")
namesRdd = names.map(parseName)

lines = sc.textFile("DataSets/Marvel-Data/Marvel-Graph.txt")

heroPairings = lines.map(countCoOccurences)
totalCoOccurencesByHero = heroPairings.reduceByKey(lambda x, y: x + y)
flippedTotalCoOccurencesByHero = totalCoOccurencesByHero.map(lambda x: (x[1], x[0]))

mostPopular = flippedTotalCoOccurencesByHero.max()

mostPopularHero = namesRdd.lookup(mostPopular[1])[0]

print(str(mostPopularHero), "is the most popular superhero, with ", str(mostPopular[0]), " co-appearances.")

