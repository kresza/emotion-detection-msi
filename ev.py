import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

model = load_model('emotion_model.h5')

train_data_dir = 'train'
emotion_labels = {i: emotion for i, emotion in enumerate(os.listdir(train_data_dir))}

face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)

img_width, img_height = 48, 48

while True:
    ret, image = video.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_img = image[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (img_width, img_height))
        face_img = face_img.astype('float32') / 255
        face_img = tf.convert_to_tensor(face_img, dtype=tf.float32)
        face_img = tf.expand_dims(face_img, axis=0)

        prediction = model.predict(face_img)
        dominant_emotion = emotion_labels[np.argmax(prediction)]

        cv2.putText(image, dominant_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('video', image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


video.release()
cv2.destroyAllWindows()
