from pyspark import SparkConf, SparkContext


def loadMovieNames():
    movieNames = {}
    with open("DataSets/ml-100k/u.ITEM", encoding="ISO-8859-1") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames


conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf=conf)


nameDict = sc.broadcast(loadMovieNames())

lines = sc.textFile("DataSets/ml-100k/u.data")
movies = lines.map(lambda x: (int(x.split()[1]), 1))
movieCounts = movies.reduceByKey(lambda x, y: x + y)
flipped = movieCounts.map(lambda x: (x[1], x[0]))
sortedMovies = flipped.sortByKey()

sortedMoviesWithNames = sortedMovies.map(lambda movie: (movie[0], nameDict.value[movie[1]]))

results = sortedMoviesWithNames.collect()

for result in results:
    print(result)
