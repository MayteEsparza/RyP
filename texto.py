import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo previamente entrenado (en formato .h5)
modelo_entrenado = load_model('C:/Users/andre/Downloads/modelo_entrenado_con_una_imagen.h5')

# Cargar el dataset
dataset_path = "C:/Users/andre/OneDrive/Documentos/tareas.csv"
tareas_df = pd.read_csv(dataset_path)

# Mostrar las primeras filas del dataset
print("Tareas disponibles:")
print(tareas_df.head())

# Función para generar una predicción basada en el modelo
def generar_respuesta(image_path):
    # Cargar y preprocesar la imagen
    img = Image.open(image_path).convert('RGB')
    img = img.resize((150, 150))  # Cambia el tamaño según lo que necesites
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión para el batch

    # Realizar la predicción
    respuesta = modelo_entrenado.predict(img_array)
    return respuesta

# Iterar sobre las tareas en el dataset
for index, row in tareas_df.iterrows():
    tarea = row['tarea']
    print(f"\nTarea: {tarea}")
    respuesta = generar_respuesta(tarea)
    print(f"Respuesta: {respuesta[0][0]}")  # Ajusta según la salida de tu modelo
