import sys
from pyspark import SparkConf, SparkContext
from math import sqrt

# Conf will be passed with the command line when run on the cluster
conf = SparkConf()
sc = SparkContext(conf=conf)

scoreThreshold = None
coOccurrenceThreshold = None
movieID = None


def loadMovieNames():
    movieNames = {}
    with open("DataSets/ml-1m/movies.dat", encoding="ascii", errors='ignore') as file:
        for line in file:
            fields = line.split('::')
            movieNames[int(fields[0])] = fields[1]
    return movieNames


def createPairs(userRatings):
    userRatingsPairs = userRatings[1]
    (movie1, rating1) = userRatingsPairs[0]
    (movie2, rating2) = userRatingsPairs[1]
    return (movie1, movie2), (rating1, rating2)


def filterDuplicates(userRatings):
    DuplicateRatings = userRatings[1]
    (movie1, rating1) = DuplicateRatings[0]
    (movie2, rating2) = DuplicateRatings[1]
    return movie1 < movie2


def computeCosineSimilarities(ratingsPair):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingsPair:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0

    if denominator:
        score = (numerator / float(denominator))

    return score, numPairs


def filterResults(pairSim):
    movie1 = pairSim[0][0]
    movie2 = pairSim[0][1]
    score = pairSim[1][0]
    occurrence = pairSim[1][1]
    if (movie1 == movieID or movie2 == movieID) and score > scoreThreshold and occurrence > coOccurrenceThreshold:
        return (movie1, movie2), (score, occurrence)


print("\n Loading movie names...")
nameDict = loadMovieNames()

data = sc.textFile("LOCATION OF RATING FILE")

# Mapping ratings to key / value pairs in format user ID => (movie ID, rating)
ratings = data.map(lambda l: l.split("::")).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))


ratingsPartitioned = ratings.partitionBy(100)
# Joining the RDD on itself to find out all combinations of movies watched and rated by the same user
joinedRatings = ratingsPartitioned.join(ratingsPartitioned)

# joinedRatings will contain RDD of user ID => ((movieID, rating), (movieID, Rating))


# Filter all the duplicates from RDD
uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)

# Create pairs on movies ((movie1, movie2), (rating1, rating2))
moviePairs = uniqueJoinedRatings.map(createPairs)

# Grouping all the rating pairs for each uniques pair of movies
moviePairRating = moviePairs.groupByKey()

# Now we have all the ratings for a given movie pair (movie1, movie2) => (rating1, rating2), (rating1, rating2)...

# Compute similarities using cosine similarity algorithm
movieSimilarities = moviePairRating.mapValues(computeCosineSimilarities).cache()
moves = movieSimilarities.collect()

if len(sys.argv) > 1:
    scoreThreshold = 0.97
    coOccurrenceThreshold = 50

    movieID = int(sys.argv[1])

    # NOTE Replaces below lambda with a UDF (User Defined Function)
    #  filterResults = movieSimilarities.filter(lambda pairSim:
    #                   (pairSim[0][0] == movieID or pairSim[0][1] == movieID)
    #                   and pairSim[1][0] > scoreThreshold
    #                   and pairSim[1][1] > coOccurrenceThreshold)

    filterResults = movieSimilarities.filter(filterResults)
#
results = filterResults.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending=False).take(10)
#
print("Top 10 similar movies for ", nameDict[movieID])
for result in results:
    (sim, pair) = result
    # Display the similarity result that isn't the movie we're looking at
    similarMovieID = pair[0]
    if similarMovieID == movieID:
        similarMovieID = pair[1]
    print(nameDict[similarMovieID], "\tscore: ", str(sim[0]), "\tstrength: ", str(sim[1]))
