# -*- coding: utf-8 -*-
"""
To Dectect Face in Emotion Recognition
Created on 4/02/2018 
By Jainam Gala
"""
import cv2
from keras.models import load_model
import numpy

face_models_path="../trained_models/face_detection_models/haarcascade_frontalface_default.xml"
emotion_models_path="../trained_models/emotion_models/emotion_recog_1_0.346562721495.model"
emotion_labels =["Angry","Fear","Happy","Sad","Surprise","Neutral"]#[] ka matter nhi krta () ka does


face_detection=cv2.CascadeClassifier(face_models_path)
emotion_models =load_model(emotion_models_path)
emotion_model_input_size =emotion_models.input_shape[1:3]

cap=cv2.VideoCapture(0)
while (True):
    ret_val,frame=cap.read()
    if ret_val==True:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
       
        faces=face_detection.detectMultiScale(gray,1.3,5)
        for x,y,w,h in faces:
                gray_face =gray[y:y+h,x:x+w]
                gray_face =cv2.resize(gray_face,emotion_model_input_size)
                pre_processed_img=gray_face.astype("float32")#32 bit representatn hoga 4r normalizatn & array operation etc***float mai convert ..coz divide by 255 toh division ho sake so
                pre_processed_img/=255
                expanded_dimen_img=numpy.expand_dims(pre_processed_img,0)#1st index
                expanded_dimen_img= numpy.expand_dims(expanded_dimen_img,-1)#last index
                emotion_probabilities=emotion_models.predict(expanded_dimen_img)
                emotion_max_prob=numpy.max(emotion_probabilities)# Not necessary argmax does the required function..arg gives index..other gives value
                emotion_label =numpy.argmax(emotion_probabilities)
                
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
                cv2.putText(frame,emotion_labels[emotion_label],(x,y),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0),5) 
                
        cv2.imshow("Emotion_recognition",frame)
#    time.sleep(0.05)
        if cv2.waitKey(1)==27:
            break
cv2.destroyAllWindows()
cap.release()            
                
                
                
        
     
