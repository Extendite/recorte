import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Asegúrate de tener este archivo en tu repo o ajusta la ruta
CASCADE_PATH = "haarcascade_frontalface_default.xml"

def detectar_rostros(image_bgr):
    """Detecta rostros usando Haar Cascade en una imagen BGR (NumPy).
       Retorna lista de (x, y, w, h)."""
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Ajusta scaleFactor y minNeighbors si deseas cambiar sensibilidad
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def recortar_cuadrado_en_torno_al_rostro(image_pil, x, y, w, h):
    """
    Toma una imagen PIL y la info de un bounding box de rostro (x, y, w, h).
    Devuelve un recorte cuadrado de la imagen (sin zoom) usando como
    centro la cara detectada.
    Lado del cuadrado = min(width, height) de la imagen.
    """
    width, height = image_pil.size

    # Centro del rostro (float)
    center_x = x + w/2
    center_y = y + h/2

    # Lado del cuadrado = el lado menor de la imagen
    side = min(width, height)

    # Coordenadas del nuevo cuadrado (top-left y bottom-right)
    left = center_x - side/2
    top = center_y - side/2
    right = left + side
    bottom = top + side

    # Evitar que el recorte quede fuera de la imagen (clamp)
    if left < 0:
        left = 0
        right = side
    if top < 0:
        top = 0
        bottom = side
    if right > width:
        right = width
        left = width - side
    if bottom > height:
        bottom = height
        top = height - side

    # Convertir a int para crop
    left, top, right, bottom = map(int, [left, top, right, bottom])

    # Recorte
    cropped = image_pil.crop((left, top, right, bottom))
    return cropped

def main():
    st.title("Recorte Cuadrado Centrado en Rostro (sin zoom)")
    st.write("Sube una imagen. Detectamos el rostro y recortamos un cuadrado centrado en él, manteniendo la escala original.")

    uploaded_file = st.file_uploader("Sube tu imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Cargamos la imagen como PIL
        image_pil = Image.open(uploaded_file).convert("RGB")

        # Convertimos a NumPy BGR para la detección con OpenCV
        image_np = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Detectamos rostros
        faces = detectar_rostros(image_bgr)

        if len(faces) == 0:
            st.warning("No se detectó ningún rostro. Mostramos la imagen original.")
            st.image(image_pil, caption="Imagen original", use_container_width=True)
        else:
            # Podrías elegir la cara más grande u ordenar las detecciones;
            # aquí simplemente tomamos la primera
            x, y, w, h = faces[0]

            # Hacemos el recorte cuadrado centrado en esa cara
            cropped_pil = recortar_cuadrado_en_torno_al_rostro(image_pil, x, y, w, h)

            st.image(cropped_pil, caption="Imagen recortada (cuadrada)", use_container_width=True)

            # Preparar para descarga
            img_bytes = io.BytesIO()
            cropped_pil.save(img_bytes, format="JPEG")
            img_bytes.seek(0)

            st.download_button(
                label="Descargar imagen recortada",
                data=img_bytes,
                file_name="imagen_cuadrada.jpg",
                mime="image/jpeg"
            )

if __name__ == "__main__":
    main()
