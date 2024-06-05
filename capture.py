import cv2
import os
from datetime import datetime

face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)

save_dir = 'captured_faces'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

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

        label = input("Enter emotion label for the captured face (happy, sad, etc.): ")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        file_path = os.path.join(save_dir, f"{label}_{timestamp}.jpg")
        cv2.imwrite(file_path, face_img)

    cv2.imshow('video', image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
