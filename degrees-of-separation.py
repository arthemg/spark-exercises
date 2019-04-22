from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf=conf)

startCharacterID = 5306  # Spider-Man ID
targetCharacterID = 14  # Adam 3,031

hitCounter = sc.accumulator(0)


def convertToBFS(line):
    fields = line.split()
    heroID = int(fields[0])
    connections = []
    for connection in fields[1:]:
        connections.append(int(connection))

    color = 'WHITE'
    distance = float('inf')

    if heroID == startCharacterID:
        color = 'GRAY'
        distance = 0
    return heroID, (connections, distance, color)


def createStartingRdd():
    inputFile = sc.textFile("DataSets/Marvel-Data/Marvel-Graph.txt")
    return inputFile.map(convertToBFS)


def bfsMap(node):
    characterID = node[0]
    data = node[1]
    connections = data[0]
    distance = data[1]
    color = data[2]

    results = []

    if color == 'GRAY':
        for connection in connections:
            newCharacterID = connection
            newDistance = distance + 1
            newColor = 'GRAY'
            if targetCharacterID == connection:
                hitCounter.add(1)

            newEntry = (newCharacterID, ([], newDistance, newColor))
            results.append(newEntry)
        color = 'Black'

    results.append((characterID, (connections, distance, color)))
    return results


def bfsReduce(data1, data2):
    edges1 = data1[0]
    edges2 = data2[0]
    distance1 = data1[1]
    distance2 = data2[1]
    color1 = data1[2]
    color2 = data2[2]

    distance = float('inf')
    color = color1
    edges = []

    if len(edges1) > 0:
        edges.extend(edges1)
    if len(edges2) > 0:
        edges.extend(edges2)

    if distance1 < distance:
        distance = distance1

    if distance2 < distance:
        distance = distance2

    if color1 == 'WHITE' and (color2 == 'GRAY' or color2 == 'BLACK'):
        color = color2

    if color1 == 'GRAY' and color2 == 'BLACK':
        color = color2

    if color2 == 'WHITE' and (color1 == 'GRAY' or color1 == 'BLACK'):
        color = color1

    if color2 == 'GRAY' and color1 == 'BLACK':
        color = color1

    return edges, distance, color


iterationRdd = createStartingRdd()

for iteration in range(0, 10):
    print("Running BFS iteration#", str(iteration + 1))

    mapped = iterationRdd.flatMap(bfsMap)

    print("Processing " + str(mapped.count()) + " values.")

    if hitCounter.value > 0:
        print("Hit the target character! From ", str(hitCounter.value), "different direction(s).")
        break

    # Reducer combines data for each character ID, preserving the darkest
    # color and shortest path.
    iterationRdd = mapped.reduceByKey(bfsReduce)
