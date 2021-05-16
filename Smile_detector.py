import cv2

face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_classifier = cv2.CascadeClassifier("haarcascade_smile.xml")

# image smile detection

image = cv2.imread("cap2.jpg")
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face = face_classifier.detectMultiScale(grayscale_image)
print(face)

for (x, y, w, h) in face:
    the_face = image[y:y+h, x:x+w]
    grayscale_face = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
    smile = smile_classifier.detectMultiScale(grayscale_face, scaleFactor=1.7, minNeighbors=20)
    print(smile)
    
    
    if len(smile)>0:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 3)
        #for (x_, y_, w_, h_) in smile:
        #   cv2.rectangle(image, (x+x_, y+y_), (x+x_+w_, y+y_+h_), (0, 0, 255), 2)
        cv2.putText(image, "Smiling", (x, y+h+60), fontScale=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 255, 0), thickness=3)

cv2.imshow("Smile Detector", image)    
cv2.waitKey()

# real time smile detection

webcam = cv2.VideoCapture(0)
while True:
    success, frame = webcam.read()
    if success:
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_classifier.detectMultiScale(grayframe)
        print(face)

        for (x, y, w, h) in face:
            the_face = frame[y:y+h, x:x+w]
            grayscale_face = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
            smile = smile_classifier.detectMultiScale(grayscale_face, scaleFactor=1.7, minNeighbors=20)
            print(smile)

            if len(smile)>0:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
                #for (x_, y_, w_, h_) in smile:
                    #cv2.rectangle(frame, (x+x_, y+y_), (x+x_+w_, y+y_+h_), (0, 0, 255), 2)
                cv2.putText(frame, "Smiling", (x, y+h+60), fontScale=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 255, 0), thickness=3)
    else:
        break

    cv2.imshow("Smile Detector", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()
            
            












            
