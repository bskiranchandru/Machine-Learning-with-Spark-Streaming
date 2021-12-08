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
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron

def multi_nb(x,y):
                    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)
        
                    # train scikit learn model 
                    MultiNB=MultinomialNB()
                    MultiNB.partial_fit(x_train,y_train,classes=[0,4])

                    f = open('MultinominalNB','wb')
                    pickle.dump(MultiNB,f)
                    f.close()

                    y_expect=y_test
                    y_pred=MultiNB.predict(x_test)
                    print(accuracy_score(y_expect,y_pred))

                    print('=====Batch Completed======')
def pass_classi(x,y):
                    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)
        
                    # train scikit learn model 
                    clf=PassiveAggressiveClassifier()
                    clf.partial_fit(x_train,y_train,classes=[0,4])

                    f1= open('PassiveAggressiveClassifier','wb')
                    pickle.dump(clf,f1)
                    f1.close()

                    y_expect=y_test
                    y_pred=clf.predict(x_test)
                    print(accuracy_score(y_expect,y_pred))

                    print('=====Batch Completed======')
def percep(x,y):
                    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)
        
                    # train scikit learn model 
                    cp=Perceptron()
                    cp.partial_fit(x_train,y_train,classes=[0,4])

                    f1= open('Perceptron','wb')
                    pickle.dump(cp,f1)
                    f1.close()

                    y_expect=y_test
                    y_pred=cp.predict(x_test)
                    print(accuracy_score(y_expect,y_pred))

                    print('=====Batch Completed======')

