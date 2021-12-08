import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
f = open('MultinominalNB','rb')
model = pickle.load(f)

f1 = open('PassiveAggressiveClassifier','rb')
model1 = pickle.load(f1)

f2= open('Perceptron','rb')
model2=pickle.load(f2)
cm = np.zeros((2,2))
cm1 = np.zeros((2,2))
cm2 = np.zeros((2,2))

def test_model_bernouli(x,y):
    #print(cm)
    global cm
    print('2')
    predictions = model.predict(x)
    
    acc = accuracy_score(y,predictions)

    cm = np.add(cm,confusion_matrix(y,predictions))
    #print('xxxxx')
    try:
        f1 = open('acc_store.txt','a')
        f1.write('%s\n'%str(acc))
        #print('yvvvv')
        f1.close()
    except Exception as e:
         print(e)
    try:
        f2 = open('cf_matrix_bernouli','w')
        lis = cm.flatten()

        for i in lis:
           f2.write('%s\n'%i) 
        f2.close()
    except Exception as e:
        print(e)

def test_model_pasc(x,y):
    global cm1
    predictions = model1.predict(x)
    acc = accuracy_score(y,predictions)
    #print('000')
    cm1 = np.add(cm1,confusion_matrix(y,predictions))
    #print('yyyy')
    f3 = open('acc_store1.txt','a')
    f3.write('%s\n'%str(acc))
    #print('tttt')
    f3.close()
    
    f4 = open('cf_matrix_bernouli1','w')
    lis = cm1.flatten()
    for i in lis:
        f4.write('%s\n'%i) 
    f4.close()
def test_model_perc(x,y):
    global cm2
    predictions = model2.predict(x)
    acc = accuracy_score(y,predictions)
    cm2 = np.add(cm2,confusion_matrix(y,predictions))
    #print('zzz')
    f2 = open('acc_store2.txt','a')
    f2.write('%s\n'%str(acc))
    f2.close()
    
    f3 = open('cf_matrix_bernouli2','w')
    lis = cm2.flatten()
    for i in lis:
       f3.write('%s\n'%i) 
    f3.close()



