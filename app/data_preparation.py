#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
#from tensorflow.keras.optimizers import Adam
import zipfile
import os
import shutil
from PIL import Image
from app.config import RUTA_ZIP, RUTA_EXTRACCION, RUTA_DATOS

def es_imagen_valida(filepath):
    """ Verifica si el archivo es una imagen válida """
    try:
        with Image.open(filepath) as img:
            img.verify()  # Verifica si la imagen es válida
        return True
    except (IOError, SyntaxError):
        return False

def limpiar_dataset():
    """ Elimina archivos corruptos o no válidos """
    directorios = ["dataset/train/cats", "dataset/train/dogs"]
    
    for directorio in directorios:
        for archivo in os.listdir(directorio):
            ruta_archivo = os.path.join(directorio, archivo)
            
            # Si no es una imagen válida, eliminarla
            if not es_imagen_valida(ruta_archivo):
                print(f"❌ Archivo no válido eliminado: {ruta_archivo}")
                os.remove(ruta_archivo)

def preparar_datos():
    """
    Extrae el archivo ZIP y mueve las imágenes a una carpeta estructurada para entrenamiento.

    :param ruta_zip: Ruta del archivo ZIP con las imágenes.
    :param ruta_extraccion: Carpeta donde se extraerán los archivos.
    :param ruta_destino: Carpeta donde se organizarán los datos para entrenamiento.
    """

    # Extraer el ZIP si aún no ha sido extraído
    if not os.path.exists(RUTA_EXTRACCION):
        os.makedirs(RUTA_EXTRACCION, exist_ok=True)
        with zipfile.ZipFile(RUTA_ZIP, 'r') as zip_ref:
            zip_ref.extractall(RUTA_EXTRACCION)
        print(f"✅ Datos extraídos en '{RUTA_EXTRACCION}'")
    else:
        print("ℹ️ Los datos ya están extraídos.")

    # Rutas originales dentro del ZIP
    ruta_base = os.path.join(RUTA_EXTRACCION, "PetImages")
    ruta_gatos = os.path.join(ruta_base, "Cat")
    ruta_perros = os.path.join(ruta_base, "Dog")

    # Rutas destino para entrenamiento
    os.makedirs(RUTA_DATOS, exist_ok=True)
    ruta_train_gatos = os.path.join(RUTA_DATOS, "cats")
    ruta_train_perros = os.path.join(RUTA_DATOS, "dogs")

    # Crear carpetas destino si no existen
    os.makedirs(ruta_train_gatos, exist_ok=True)
    os.makedirs(ruta_train_perros, exist_ok=True)

    # Mover archivos de gatos
    for archivo in os.listdir(ruta_gatos):
        ruta_archivo = os.path.join(ruta_gatos, archivo)
        if os.path.isfile(ruta_archivo):  # Solo mover archivos
            shutil.move(ruta_archivo, os.path.join(ruta_train_gatos, archivo))

    # Mover archivos de perros
    for archivo in os.listdir(ruta_perros):
        ruta_archivo = os.path.join(ruta_perros, archivo)
        if os.path.isfile(ruta_archivo):  # Solo mover archivos
            shutil.move(ruta_archivo, os.path.join(ruta_train_perros, archivo))

    print(f"✅ ¡Archivos organizados correctamente en '{RUTA_DATOS}'!")
    limpiar_dataset()