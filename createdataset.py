import cv2
import numpy as np
import os

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def show_webcam(mirror=False):
    id = str(raw_input("enter the id"))
    os.makedirs("dataSet/"+id)
    cam = cv2.VideoCapture(1)
    sample = 0
    while True:
        ret_val, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sample = sample + 1
            cv2.imwrite("dataSet/"+id+"/User."+str(id)+"."+str(sample)+".png", gray[y:y+h, x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.waitKey(100)
        cv2.imshow('my webcam', img)
        if sample > 20:
            break
        if mirror: 
            img = cv2.flip(img, 1)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cam.release()
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


main()
