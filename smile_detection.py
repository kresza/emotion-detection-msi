from deepface import DeepFace
import cv2

face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)
while True:
    ret, image = video.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors= 5,
        minSize= (30,30)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
        face_img = image[y:y+h, x:x+w]  # Wykadrowanie twarzy z obrazu
        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        print("Result:", result)
        if result:
            dominant_emotion = result[0]['dominant_emotion']
            cv2.putText(image, dominant_emotion, (x,y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255,0), 3, cv2.LINE_AA)
    cv2.imshow('video', image)
    k = cv2.waitKey(30) & 0xff

    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
