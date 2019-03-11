# -*- coding: utf-8 -*-

#step1: is to check your dataset properly coz it may cause errors
"""if tensflow not downloading or something..backend change to theano

from keras import backend as K
import importlib
import os 

def set_keras_backend(backend):
    if K.backend() !=backend:
        os.environ["KERAS_BACKEND"]=backend
        importlib.reload(K)
        assert K.backend()==backend
        
set_keras_backend("theano")                    
        
in c drive-keras.json...mai tensorflow ko theano change backend
"""

#import keras
#from keras import backend as K 
import os
import cv2
import numpy
#from os.path import join
#import random
from itertools import repeat
from cnn_model import buildCNNModel
num_classes=6

def listAllFilesPath(path,formats=["png","jpg","jpeg","tif"]):#folder se file lena hai in that format only which is 
    results=[]
    for root,subFolders,files in os.walk(path):
        for file in files:
            if file.split(".")[-1] in formats:#last ka extentn batayega..jpg etc coz of -1
                results.append("/".join([root,file]))#join karega file name ..
    return results

def preProcessImage(path,img_width,img_height):
    img=cv2.imread(path,0)#grayscale mai convert karna hai
    img=cv2.resize(img,(img_width,img_height),
                   interpolation=cv2.INTER_CUBIC)#pading karega taki extenstn mai problem na ho usko
    img=img.astype("float32")#32 bit representatn hoga 4r normalizatn & array operation etc***float mai convert ..coz divide by 255 toh division ho sake so
    img/=255
    return img

#return karega 2 value array mai
def prepareData(size):
    input_samples=[]
    output_labels=[]
    for _class in range(0,num_classes):
        path="../dataset/ferDataset/%d" %(_class)#class ka value append karega eg :0
        length=len(os.listdir(path))#folder mai images ka milega length
        samples=numpy.array(list(map(preProcessImage,listAllFilesPath(path),repeat(size[0],length),repeat(size[1],length))))
        input_samples.append(samples)
        output_labels.append(numpy.array([_class]*len(samples)))
    
        #comments ka bhi indentation matter krta hai uppar wale statement jese hi hona chahiye
        """numpy.array([0]*10)
        Out[2]: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])          
        
                
        inputs=[]
        
        inputs.append(numpy.array([0]*10))
        
        inputs.append(numpy.array([1]*10))
        
        
        F5 se autosave n run 
        """        
        
    inputs=numpy.concatenate(input_samples,axis=0)
    outputs=numpy.concatenate(output_labels,axis=0)
    
    #koi bhi functn ka op chahiye toh console mai put that ..    
    
    #convert to hot vectors 
    output_hot_vectors=numpy.zeros((len(outputs),num_classes))
    output_hot_vectors[numpy.arange(len(outputs)),outputs]=1
    outputs=output_hot_vectors
    
    #shuffle the inputs and outputs same way
    p=numpy.random.permutation(len(inputs))#p=numpy.random.permutation(10)
    inputs=inputs[p]
    outputs=outputs[p]
    
    return inputs,outputs

if __name__=="__main__": #main loop
    no_of_epochs=1
    emotion_models_path="../trained_models/emotion_models/"
    size=[64,64] #256 etc..kaam so kaam time lagega 
    inputs,outputs=prepareData(size)
    inputs=inputs.reshape(inputs.shape[0],inputs.shape[1],inputs.shape[2],1)#face to single detect among all the samples given
    
    num_of_samples=len(inputs)
    train_data_length=int(num_of_samples*0.80)#no of samples giving for training data
    x_train,x_test=inputs[0:train_data_length],inputs[train_data_length:]#x inputs.. y outputs
    y_train,y_test=outputs[0:train_data_length],outputs[train_data_length:]
    
    #architecture defined
    model=buildCNNModel(inputs.shape[1:],num_classes,32,(3,3),0.05,(2,2),1)
    print(model.summary())
    
    #training model
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    history=model.fit(x_train,y_train,batch_size=16,epochs=no_of_epochs,validation_data=(x_test,y_test))#batch_size kaam or else CPU heated
    
    
    if not os.path.exists(emotion_models_path):
        os.makedirs(emotion_models_path) 
    model.save(emotion_models_path + 'emotion_recog_%d_%s.model' %(no_of_epochs,history.history["val_acc"][0]))
    
        
        
        
        
        
        
        
        
        
    
    
    