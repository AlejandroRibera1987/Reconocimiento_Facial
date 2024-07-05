import cv2
import numpy as np
import os
from PIL import Image
# Crear el reconocedor LBPHFaceRecognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Clasificador Haar para la detección de rostros
detector = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

# Función para obtener imágenes y etiquetas
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # convertir a escala de grises
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

# Ruta para la base de datos de imágenes de rostros
path = 'dataset'

print("\n [INFO] Entrenando rostros. Tomará unos segundos. Espere ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Guardar el modelo entrenado
recognizer.save('trainer/trainer.yml')

# Imprimir el número de rostros entrenados y finalizar el programa
print("\n [INFO] {0} rostros entrenados. Programa finalizado".format(len(np.unique(ids))))
