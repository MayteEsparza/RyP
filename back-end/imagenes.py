import pandas as pd
from diffusers import StableDiffusionPipeline

# Cargar el modelo de generación de imágenes
model_id = "CompVis/stable-diffusion-v1-4"  # O cualquier otro modelo disponible
pipeline = StableDiffusionPipeline.from_pretrained(model_id)
pipeline = pipeline.to("cuda")  # Usa "cuda" si tienes una GPU disponible

# Cargar el dataset
dataset_path = "tareas.csv"
tareas_df = pd.read_csv(dataset_path)

# Mostrar las primeras filas del dataset
print("Tareas disponibles:")
print(tareas_df.head())

# Función para generar imágenes a partir de una descripción
def generar_imagen(descripcion, index):
    image = pipeline(descripcion).images[0]
    image.save(f"imagen_{index}.png")  # Guarda la imagen generada

# Iterar sobre las tareas en el dataset
for index, row in tareas_df.iterrows():
    tarea = row['tarea']
    print(f"\nTarea: {tarea}")
    
    # Generar imagen basada en la tarea
    generar_imagen(tarea, index)
    print(f"Imagen guardada como: imagen_{index}.png")

# Notificación de finalización
print("Generación de imágenes completada.")
