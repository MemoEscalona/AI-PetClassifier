import matplotlib.pyplot as plt  # ✅ Agregar esta línea
from app.model import construir_modelo
from app.data_preprocessing import preprocesar_datos
from app.config import RUTA_MODELO, EPOCHS
import os

def entrenar_modelo():
    # Preparar los datos
    print("🔄 Preparando datos...")
    train_generator, val_generator = preprocesar_datos()

    # Construir el modelo
    print("🛠️ Construyendo modelo...")
    modelo = construir_modelo()

    # Entrenar el modelo
    print("🏋️‍♂️ Iniciando entrenamiento...")
    historial = modelo.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

    # Evaluar el modelo en el conjunto de validación
    print("📊 Evaluando modelo...")
    pérdida, precisión = modelo.evaluate(val_generator)
    print(f"📊 Evaluación del modelo - Pérdida: {pérdida:.4f}, Precisión: {precisión:.4f}")

    # Guardar el modelo entrenado
    print(f"💾 Guardando modelo en {RUTA_MODELO}...")
    modelo.save(RUTA_MODELO)
    print(f"✅ Modelo guardado en '{RUTA_MODELO}'")

    print("📈 Generando gráficos de entrenamiento...")
    graficar_historial(historial)
    print("✅ ¡Entrenamiento completado!")

def graficar_historial(historial):
    print("Historial del entrenamiento:", historial.history)
    """ Genera gráficos de pérdida y precisión del entrenamiento """
    plt.figure(figsize=(12, 5))

    # Gráfico de Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(historial.history['loss'], label='Pérdida Entrenamiento')
    plt.plot(historial.history['val_loss'], label='Pérdida Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Evolución de la Pérdida')
    plt.legend()
    plt.ylim(0, 1)  # Fija los valores entre 0 y 1 para evitar problemas de escalado
    plt.xlim(0, EPOCHS)  # Usa el número total de épocas en el eje X


    # Gráfico de Precisión
    plt.subplot(1, 2, 2)
    plt.plot(historial.history['accuracy'], label='Precisión Entrenamiento')
    plt.plot(historial.history['val_accuracy'], label='Precisión Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.title('Evolución de la Precisión')
    plt.legend()
    plt.ylim(0, 1)  # Fija los valores entre 0 y 1 para evitar problemas de escalado
    plt.xlim(0, EPOCHS)  # Usa el número total de épocas en el eje X

    # Guardar la gráfica en un archivo dentro del contenedor
    ruta_guardado = "/app/models/historial_entrenamiento.png"
    os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
    plt.savefig(ruta_guardado)
    print(f"✅ Gráfica guardada en {ruta_guardado}")
# Si se ejecuta directamente este archivo, se entrena el modelo
if __name__ == "__main__":
    entrenar_modelo()