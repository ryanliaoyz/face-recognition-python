import cv2
import numpy as np
import os, os.path

fileCount = len(os.listdir('trainer/'))
recognizer=[0] * fileCount
Id = [0] * fileCount
conf = [0] * fileCount
dbfile  = open('db.txt', 'r')
lineCount = 0
lineContent = [0] * 300
dbname = [0]*300
dbId = [0]*300
print(fileCount, "file")

for line in dbfile:
    lineContent[lineCount] = line
    dbId[lineCount] = lineContent[lineCount].split()[0]
    dbname[lineCount] = lineContent[lineCount].split()[1]
    print(dbId[lineCount], dbname[lineCount])
    lineCount = lineCount + 1




for i in range(0, fileCount):
    print("cure",i)
    recognizer[i] = cv2.face.LBPHFaceRecognizer_create()
    recognizer[i].read('trainer/{a}.yml'.format(a = i))

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
        for i in range(0, fileCount):
            Id[i], conf[i] = recognizer[i].predict(gray[y:y+h,x:x+w])
            print(Id[i], conf[i], "aaaaaaaaaa")
        minConf = conf[0]
        minId = Id[0]
        for j in range(0, fileCount):
            print(Id[j], conf[j], "bbbbbbbbbbb")
            if conf[j] < minConf:
                minConf = conf[j]
                minId = Id[j]
        print(minId, minConf)
        
        cv2.putText(im, dbname[minId - 1], (x,y+h), fontface, fontscale, fontcolor)
    cv2.imshow('face',im) 
    if cv2.waitKey(10) == ord('q'):
        break
    if cv2.waitKey(1) == 27: 
        break  # esc to quit
cam.release()
cv2.destroyAllWindows()