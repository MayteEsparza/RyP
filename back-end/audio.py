import pandas as pd
from gtts import gTTS
import os

# Cargar el dataset
dataset_path = "C:/Users/andre/OneDrive/Documentos/tareas.csv"
tareas_df = pd.read_csv(dataset_path)

# Mostrar las primeras filas del dataset
print("Tareas disponibles:")
print(tareas_df.head())

# Función para generar audio a partir de texto
def generar_audio(texto, nombre_archivo):
    tts = gTTS(text=texto, lang='es')
    tts.save(nombre_archivo)

# Simulación de generación de respuestas
def generar_respuesta(tarea):
    # Aquí puedes implementar tu modelo de IA para generar respuestas
    # Por simplicidad, usaremos un texto fijo
    return f"Esta es la respuesta a la tarea: {tarea}"

# Iterar sobre las tareas en el dataset
for index, row in tareas_df.iterrows():
    tarea = row['tarea']
    print(f"\nTarea: {tarea}")
    respuesta = generar_respuesta(tarea)
    print(f"Respuesta: {respuesta}")

    # Generar audio para la respuesta
    nombre_archivo = f"respuesta_{index}.mp3"
    generar_audio(respuesta, nombre_archivo)
    print(f"Audio guardado como: {nombre_archivo}")

# Notificación de finalización
print("Generación de audio completada.")
