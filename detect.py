import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/dataset.yml')
#recognizer.read()
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


cam = cv2.VideoCapture(1)
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        print(len(Id))
        if(conf < 102):
            if(int(Id) == 1):
                id="Ryan{a} {b}".format(a = Id, b = conf)
            elif(int(Id) == 2):
                id="Oliver{a} {b}".format(a = Id, b = conf)
            elif(int(Id)== 3):
                id="Howard{a} {b}".format(a = Id, b = conf)
        else:
            id="Unknown{a} {b}".format(a = Id, b = conf)
        cv2.putText(im, id, (x,y+h), fontface, fontscale, fontcolor)
    cv2.imshow('face',im) 
    if cv2.waitKey(10) == ord('q'):
        break
    if cv2.waitKey(1) == 27: 
        break  # esc to quit
cam.release()
cv2.destroyAllWindows()