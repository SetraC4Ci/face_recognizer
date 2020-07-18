import cv2
import numpy as np
import os

name = str(input("name of the person : "))

# create the train image folder for the user
train_path = os.path.join("Images", "train")
if not os.path.exists(os.path.join(train_path, name)):
    os.mkdir(str(train_path) + "/" + name)
train_path = os.path.join(train_path, name)

# create the test image folder for the user
test_path = os.path.join("Images", "test")
if not os.path.exists(os.path.join(test_path, name)):
    os.mkdir(str(test_path) + "/" + name)
test_path = os.path.join(test_path, name)

face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
# print(face_cascade)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 250)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 250)

count = 0
while(True):
    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    # faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi = frame[y:y+h, x:x+w] # region of interest [ycood_start, ycoord_end]
        count += 1
        if count < 80:
            img_item = os.path.join(train_path, str(count) + ".jpg")
            cv2.imwrite(img_item, roi)
        else:
            img_item = os.path.join(test_path, str(count) + ".jpg")
            cv2.imwrite(img_item, roi)
        color = (255, 0, 0) # BGR - color for rectangle
        end_coord_x = x + w
        end_coord_y = y + h
        stroke = 2
        cv2.rectangle(frame, (x, y), (end_coord_x, end_coord_y), color, stroke)
        cv2.putText(frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, color, stroke)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 13 or count == 100:
        print(count)
        break

cap.release()
cv2.destroyAllWindows()
