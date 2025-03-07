import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Asegúrate de tener este archivo en la misma carpeta o ajusta la ruta
CASCADE_PATH = "haarcascade_frontalface_default.xml"

def detectar_rostros(image_bgr):
    """Recibe imagen BGR (NumPy) y detecta rostros con Haar Cascade.
       Retorna lista de (x, y, w, h)."""
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Ajusta parámetros a conveniencia
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def recorte_cuadrado_pillow(image_pil, bbox):
    """
    Toma una imagen PIL y el bounding box (x, y, w, h) en coordenadas.
    Retorna un recorte cuadrado PIL (centrado en el rostro).
    """
    x, y, w, h = bbox
    width, height = image_pil.size

    # Lado del cuadrado
    side = max(w, h)

    # Centro del rectángulo
    center_x = x + w/2
    center_y = y + h/2

    # Coordenadas del nuevo cuadrado
    new_x = center_x - side/2
    new_y = center_y - side/2
    new_x2 = new_x + side
    new_y2 = new_y + side

    # Asegurarnos de no salirnos de la imagen (clamp)
    # OJO: Si la cara está en un borde, quizá el cuadrado no pueda ser tan grande.
    if new_x < 0:
        new_x = 0
        new_x2 = side
    if new_y < 0:
        new_y = 0
        new_y2 = side
    if new_x2 > width:
        new_x2 = width
        new_x = width - side
    if new_y2 > height:
        new_y2 = height
        new_y = height - side

    # Convertir a int (Pillow no admite float en crop)
    new_x, new_y, new_x2, new_y2 = map(int, [new_x, new_y, new_x2, new_y2])

    # Recortar
    cropped = image_pil.crop((new_x, new_y, new_x2, new_y2))
    return cropped

def main():
    st.title("Recortador Cuadrado con Detección de Rostro")
    st.write("Sube una imagen, se detecta el rostro y se recorta un cuadrado alrededor de él (400x400).")

    uploaded_file = st.file_uploader("Sube tu imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Cargar la imagen como PIL
        image_pil = Image.open(uploaded_file).convert("RGB")

        # Convertir a NumPy BGR para pasar al detector de rostros
        image_np = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Detectamos rostros
        faces = detectar_rostros(image_bgr)

        if len(faces) == 0:
            st.write("No se detectaron rostros. Se mostrará la imagen original.")
            st.image(image_pil, caption="Imagen original", use_column_width=True)
        else:
            st.write(f"Se detectaron {len(faces)} rostro(s). Tomaremos el primero (o uno).")

            # Podrías elegir el rostro más grande o el primero. Aquí tomamos el primero.
            x, y, w, h = faces[0]

            # Obtenemos un recorte cuadrado centrado en el rostro
            cropped_pil = recorte_cuadrado_pillow(image_pil, (x, y, w, h))

            # Redimensionar a 400x400
            final_size = 400
            cropped_resized = cropped_pil.resize((final_size, final_size))

            st.image(cropped_resized, caption="Recorte cuadrado (400x400)", use_column_width=False)

            # Permitir la descarga
            img_bytes = io.BytesIO()
            cropped_resized.save(img_bytes, format="JPEG")
            img_bytes.seek(0)

            st.download_button(
                label="Descargar imagen recortada",
                data=img_bytes,
                file_name="rostro_cuadrado.jpg",
                mime="image/jpeg"
            )


if __name__ == "__main__":
    main()
