import cv2
import os

cam = cv2.VideoCapture(1)
cam.set(3, 1540) # set video width
cam.set(4, 680) # set video height

face_detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n ingrese la identificación del usuario y presione <Enter>  ')

print("\n [INFO] Inicializando la captura de rostros. Mira a la cámara y espere...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 8)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Saliendo del Programa...")
cam.release()
cv2.destroyAllWindows()