import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from PIL import Image
import pytesseract

# Indicar la ruta de Tesseract en tu sistema
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tessdata_dir_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'

# Definir el tamaño de las imágenes
img_width, img_height = 150, 150

# Cargar y preprocesar las imágenes (aquí solo un ejemplo con una imagen)
# En un caso real, deberías tener un conjunto de datos con múltiples imágenes y etiquetas
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_width, img_height))
    img_array = np.array(img) / 255.0
    return img_array

# Cargar una imagen de entrenamiento y su etiqueta
image_path = r'C:\Users\andre\Downloads\cuento.png'  # Cambia esto a tu imagen
img_array = load_and_preprocess_image(image_path)

# Simular una etiqueta (0 o 1 para clasificación binaria)
label = np.array([1])  # Cambia esto según la clase de la imagen

# Crear el modelo
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Para clasificación binaria

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo con la imagen única
model.fit(np.array([img_array]), label, epochs=10)

# Guardar el modelo entrenado en formato h5
model.save('modelo_entrenado.h5')

print("Modelo guardado como 'modelo_entrenado.h5'.")



