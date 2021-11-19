'''from pyspark import SparkContext
from pyspark.streaming import StreamingContext

# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext("local[2]", "NetworkWordCount")
ssc = StreamingContext(sc, 1)
lines = ssc.socketTextStream("localhost", 9999)
words = lines.flatMap(lambda line: line.split(" "))
pairs = words.map(lambda word: (word, 1))
wordCounts = pairs.reduceByKey(lambda x, y: x + y)

# Print the first ten elements of each RDD generated in this DStream to the console
wordCounts.print()'''
import sys
from pyspark.sql import SparkSession 
from pyspark import SparkContext
from pyspark.streaming import StreamingContext


sc = SparkContext.getOrCreate()
spark=SparkSession(sc)
sc.setLogLevel('OFF')
ssc = StreamingContext(sc, 1)

stream_data=ssc.socketTextStream("localhost",6100)
def readMyStream(rdd):
    df=spark.read.json(rdd)
    df.printSchema()
    #print('here')
    df.show()

#stream_data.pprint()
stream_data.foreachRDD(lambda x:readMyStream(x))

ssc.start()
ssc.awaitTermination()
