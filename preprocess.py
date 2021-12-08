import sys
from pyspark.sql.functions import lower,regexp_replace,col
import re
from pyspark.sql.types import*
from pyspark.sql.functions import udf
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from pyspark.ml.feature import StopWordsRemover,Tokenizer,HashingTF
from pyspark.mllib.util import MLUtils
from operator import attrgetter
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import ast
import numpy as np
from nltk.stem import PorterStemmer
ps=PorterStemmer()

def pre_process(df):   
    urlPattern =r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    alphaPattern=r"[^a-zA-Z\s]"
    userPa='@[^\s]+'
    seqPa=r"(.)\1\1+"
    seqReplacepat=r"\1\1"
    a=lambda x:[ps.stem(i.strip()) for i in ast.literal_eval(str(x))]
    
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
    
    tokenizer=Tokenizer(inputCol='tweet',outputCol='c_tweet')
    #tokenizer.setInputCol(df[tweet])
    data = tokenizer.transform(df).select('senti','c_tweet')
    stop_words=StopWordsRemover(inputCol='c_tweet',outputCol='o_tweet')
    new_data= stop_words.transform(data).select('senti','o_tweet')
    new_data=new_data.withColumn('o_tweet',udf(a,ArrayType(StringType()))('o_tweet'))
    new_data.show()

    hashTF = HashingTF(inputCol=stop_words.getOutputCol(), outputCol="features")
    numericTrainData = hashTF.transform(new_data).select('senti', 'o_tweet', 'features')
    #print('aaaaaaaaa')
    def convert_to_matrix(vec):
        data,indices = vec.values, vec.indices
        shape = 1,vec.size
        return csr_matrix((data,indices,np.array([0,vec.values.size])),shape)

    x =  numericTrainData.select('features')
    #print('bbbbbbbbbb')
    features = x.rdd.map(attrgetter('features'))
    mats = features.map(convert_to_matrix)
    mat = mats.reduce(lambda x,y: vstack ([x,y]))
    mat = np.array(mat.todense())
    
    x = mat
    y = numericTrainData.select(numericTrainData.senti.cast(IntegerType()))
    y = np.array(y.collect()).T
    y = np.array(y[0])

    return x,y

        