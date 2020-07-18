
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import PIL.Image as Image
from os import listdir

IMAGE_SIZE = 120

model = load_model("face_recognizer_model.h5")

face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml") # pylint: disable=no-member
cap = cv2.VideoCapture(0) # pylint: disable=no-member
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 250) # pylint: disable=no-member
p_list = listdir("./Images/test/")
count = 0
while(True):
    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    # faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi = frame[y:y+h, x:x+w] # region of interest [ycood_start, ycoord_end]
        face = cv2.resize(roi, (IMAGE_SIZE, IMAGE_SIZE)) # pylint: disable=no-member
        img = Image.fromarray(face, 'RGB')
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        print(prediction)
        index = np.argmax(prediction)
        name = p_list[index]

        color = (255, 0, 0) # BGR - color for rectangle
        end_coord_x = x + w
        end_coord_y = y + h
        stroke = 2
        cv2.rectangle(frame, (x, y), (end_coord_x, end_coord_y), color, stroke) # pylint: disable=no-member
        cv2.putText(frame, str(name), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, color, stroke)

    cv2.imshow('frame', frame) # pylint: disable=no-member
    cv2.waitKey(1)
