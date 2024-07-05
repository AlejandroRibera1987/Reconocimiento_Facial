import cv2

try:
    import cv2.face
    print("cv2.face disponible.")
except AttributeError:
    print("No se pudo importar cv2.face.")
