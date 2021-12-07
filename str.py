import sys
from pyspark.sql import SparkSession 
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import lower,regexp_replace,col
import re
from pyspark.sql.types import*
from pyspark.sql.functions import udf
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from pyspark.ml.feature import StopWordsRemover,Tokenizer,HashingTF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score as score
#from sklearn import metrics
import numpy as np
from sklearn.naive_bayes import BernoulliNB
#import neattext.functions as  nfx
sc = SparkContext.getOrCreate()
spark=SparkSession(sc)
sc.setLogLevel('OFF')
ssc = StreamingContext(sc, 1)

stream_data=ssc.socketTextStream("localhost",6100)
"""
def cleanTxt(text):
    text=re.sub(r'@[A-Za-z0-9]+', '',text)
    text=re.sub(r'#','',text)
    text=re.sub(r'RT[\s]+','',text)
    text=re.sub(r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)", '',text)

    return text
"""

def readMyStream(rdd):
    df=spark.read.json(rdd)
    print("a")
    #df = df.select("Senti", "Tweet").map(lambda r: LabeledPoint(r[1], [r[0]])).toDF()
    urlPattern =r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    #print("b")
    alphaPattern=r"[^a-zA-Z\s]"
    userPa='@[^\s]+'
    seqPa=r"(.)\1\1+"
    seqReplacepat=r"\1\1"
    


    user=lambda x:re.sub(userPa,'USER',x)
    sequence=lambda x:re.sub(seqPa,seqReplacepat,x)
    spaces=lambda x:re.sub(r'\s\s+','',x,flags=re.I)
    quotes=lambda x:re.sub(r'"','',x)
    alpha=lambda x:re.sub(alphaPattern,'',x)
    url=lambda x: re.sub(urlPattern,'URL',x)
    df=df.withColumn('tweet',udf(alpha,StringType())('tweet'))
    df=df.withColumn("tweet",udf(url,StringType())("tweet"))
    df=df.withColumn("tweet",udf(user,StringType())("tweet"))
    df=df.withColumn("tweet",udf(sequence,StringType())("tweet"))
    df=df.withColumn("tweet",udf(spaces,StringType())("tweet"))
    df=df.withColumn("tweet",udf(quotes,StringType())("tweet"))
    
    
    

    #stop_words = stopwords.words('english')
    #stop=lambda x:re.sub(stop_words,'',x)
    #df=df.withColumn('tweet',udf(stop,StringType())('tweet'))
    #df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x if word not in stop_words]))
    try:
        
        #print('x')
        #df=df.StopWordsRemover(stopWords=stop_words)
       
        tokenizer=Tokenizer(inputCol='tweet',outputCol='c_tweet')
        #tokenizer.setInputCol(df[tweet])
        data = tokenizer.transform(df).select('senti','c_tweet')
        stop_words=StopWordsRemover(inputCol='c_tweet',outputCol='o_tweet')
        new_data= stop_words.transform(data).select('senti','o_tweet')
        #new_data.show()
        hashTF = HashingTF(inputCol=stop_words.getOutputCol(), outputCol="features")
        numericTrainData = hashTF.transform(new_data).select('senti', 'o_tweet', 'features')
        #numericTrainData.show(truncate=False)
        train_dataset,test_dataset=numericTrainData.randomSplit([0.7,0.3])
        #train_dataset.describe().show()
        """
        LinReg=LinearRegression(featuresCol="features",labelCol="senti")
        model=LinReg.fit(train_dataset)
        pred=model.evaluate(test_dataset)
        pred.predictions.show()"""
        
        lr = LogisticRegression(labelCol="senti", featuresCol="features", maxIter=10, regParam=0.01)
        model = lr.fit(train_dataset)
        print ("Training is done!")
        prediction = model.transform(test_dataset)
        predictionFinal = prediction.select("o_tweet", "prediction", "senti")
        predictionFinal.show(truncate = False)
        correctPrediction = predictionFinal.filter(predictionFinal['prediction'] == predictionFinal['senti']).count()
        totalData = predictionFinal.count()
        print("correct prediction:", correctPrediction, ", total data:", totalData, ", accuracy:", correctPrediction/totalData)
        
        """
        x=numericTrainData.select('features').collect()
        #=x.reshape(x.shape[0]*x.shape[1],x.shape[2])
        y=numericTrainData.select('senti').collect()
        #y=np.array(y)
        #print(len(x))
        #print(len(y))
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)
        
        # train scikit learn model 
        BernNB=BernoulliNB(binarize=True)
        BernNB.fit(x_train,y_train)
        print(BernNB)
        y_expect=y_test
        y_pred=BernNB.predict(x_test)
        print(score(y_expect,y_pred))
        """
        """
        clf = LogisticRegression()
        clf.fit(x_train,y_train)
        predictions=clf.predict(x_test)
        score=clf.score(x_test,y_test)
        print ('score Scikit learn: ', score)
        """
        
        

        
    except Exception as e:
        print(e)





    
    

   

  
    
stream_data.foreachRDD(lambda x:readMyStream(x))


import time
ssc.start()
time.sleep(500)
#ssc.awaitTermination()

