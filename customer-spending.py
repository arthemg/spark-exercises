from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("Customer Spending")
sc = SparkContext(conf=conf)

def parseLine(line):
    fields = line.split(',')
    customerID = fields[0]
    payment = float(fields[2])
    return customerID, payment


lines = sc.textFile("DataSets/customer-orders.csv")
rdd = lines.map(parseLine)
totalSpent = rdd.reduceByKey(lambda x, y: x + y)
sortedPayments = totalSpent.map(lambda x: (x[1], x[0])).sortByKey()

results = sortedPayments.collect()
for result in results:
    print(result[1] + "\t{:.2f}".format(result[0]))
