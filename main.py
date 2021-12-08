import sys
from pyspark.sql import SparkSession 
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import lower,regexp_replace,col
import re
from pyspark.sql.types import*
from pyspark.sql.functions import udf
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from pyspark.ml.feature import StopWordsRemover,Tokenizer,HashingTF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
#from sklearn import metrics
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from pyspark.mllib.util import MLUtils
from operator import attrgetter
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import ast
import numpy as np
from test import *
from train import *
from preprocess import *
#import neattext.functions as  nfx
sc = SparkContext.getOrCreate()
spark=SparkSession(sc)
sc.setLogLevel('OFF')
ssc = StreamingContext(sc, 1)

stream_data=ssc.socketTextStream("localhost",6100)


def readMyStream(rdd):
    df=spark.read.json(rdd)
    x,y = pre_process(df)
    #multi_nb(x,y)
    #pass_classi(x,y)
    #percep(x,y)
    test_model_bernouli(x,y)
    test_model_pasc(x,y)
    test_model_perc(x,y)
    #print("a")
    #df = df.select("Senti", "Tweet").map(lambda r: LabeledPoint(r[1], [r[0]])).toDF()
    
       
    
stream_data.foreachRDD(lambda x:readMyStream(x))


import time
ssc.start()
time.sleep(5009)
#ssc.awaitTermination()
