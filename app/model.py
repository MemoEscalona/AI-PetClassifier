from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout  # ✅ Agregamos Dropout
from tensorflow.keras.optimizers import Adam
from app.config import TAMAÑO_IMAGEN

def construir_modelo():
    """ Construye y compila el modelo CNN. """
    modelo = Sequential([
        # Capa convolucional 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(TAMAÑO_IMAGEN[0], TAMAÑO_IMAGEN[1], 3)),
        MaxPooling2D((2, 2)),

        # Capa convolucional 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Capa convolucional 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Aplanar las capas para conectarlas a la red densa
        Flatten(),
        
        # Capa densa oculta con Dropout
        Dense(512, activation='relu'),
        Dropout(0.5),  # Ayuda a evitar overfitting
        
        # Capa de salida (1 neurona porque es clasificación binaria)
        Dense(1, activation='sigmoid')
    ])

    # Compilar el modelo
    modelo.compile(
        optimizer=Adam(learning_rate=0.0001),  # Optimización con Adam
        loss='binary_crossentropy',  # Función de pérdida para clasificación binaria
        metrics=['accuracy']  # Métrica de precisión
    )
    print("✅ Modelo compilado.")
    return modelo