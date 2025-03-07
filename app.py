import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ARCHIVO XML DEL CLASIFICADOR HAAR
CASCADE_PATH = "haarcascade_frontalface_default.xml"

def detectar_rostros(image):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def recortar_imagen(image, x, y, w, h):
    return image[y:y+h, x:x+w]

def main():
    st.title("Recortador de Imágenes (Detector de Rostros)")
    uploaded_file = st.file_uploader("Sube tu imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file).convert("RGB"))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces = detectar_rostros(image_bgr)
        if len(faces) > 0:
            st.write(f"Se han detectado {len(faces)} rostro(s).")
            for i, (x, y, w, h) in enumerate(faces):
                recorte = recortar_imagen(image_bgr, x, y, w, h)
                recorte_rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
                st.image(recorte_rgb, caption=f"Rostro #{i+1}")
        else:
            st.write("No se detectaron rostros.")

if __name__ == "__main__":
    main()
