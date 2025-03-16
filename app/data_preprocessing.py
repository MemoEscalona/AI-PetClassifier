from tensorflow.keras.preprocessing.image import ImageDataGenerator
from app.config import RUTA_DATOS, TAMAÑO_IMAGEN, BATCH_SIZE
from app.data_preparation import preparar_datos


def preprocesar_datos():
    """ Preprocesa los datos: normalización y división en entrenamiento/validación. """
    preparar_datos()
    
    #Se preprocesa la imagen aplicando algunas transformaciones
    datagen = ImageDataGenerator(
        rescale=1./255, #Normalización de pixeles, convierte los valores 0-255 a 0-1
        validation_split=0.2, #Se dividen los datos en 80% entrenamiento, 20% validación
        horizontal_flip=True, #Algunas imagenes se invierten para aumentar la diversidad del dataset
        zoom_range=0.2,  # Se realiza un zoom aletorio para mejorar la generalizacion
        shear_range=0.2 # Aplica distorsión "Shearing" esplaza una parte de la imagen en una dirección mientras mantiene la otra parte fija, causando un efecto de inclinación o deformación.
    )


    train_generator = datagen.flow_from_directory(
        RUTA_DATOS,#Se indica la ruta de origen de las imagenes 
        target_size=TAMAÑO_IMAGEN, #Ajusta las imagenes a un solo tamaño
        batch_size=BATCH_SIZE, #Agrupa las imagenes en lotes
        class_mode='binary', #Asigna el etiquetado binario, solo hay 2 clases perros y gatos
        subset='training' # Usa el 80% de las imagenes
    )

    val_generator = datagen.flow_from_directory(
        RUTA_DATOS, 
        target_size=TAMAÑO_IMAGEN, 
        batch_size=BATCH_SIZE, 
        class_mode='binary', 
        subset='validation'# Usa el 20% de las imagenes
    )

    print("✅ Generadores de datos listos.")
    return train_generator, val_generator