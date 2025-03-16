# 🐱🐶 Cats vs Dogs - Clasificación de Imágenes con Machine Learning

Este proyecto usa **TensorFlow y Flask** para entrenar una red neuronal que clasifica imágenes de **perros y gatos**.

## 🚀 Requisitos
- Python 3.8+
- TensorFlow 2.x
- Docker (opcional, para despliegue)

## 🏋️‍♂️ Entrenar el Modelo
Para entrenar el modelo, ejecuta:
```bash
python main.py --train
```
Si usas Docker:
```bash
docker run --rm -v $(pwd)/models:/app/models cats-vs-dogs python main.py --train
```

## 🌍 Ejecutar el Servidor Web
Para lanzar la API Flask:
```bash
python main.py --serve
```
Con Docker:
```bash
docker run -p 5000:5000 -v $(pwd)/models:/app/models cats-vs-dogs python main.py --serve
```

## 🐳 Construir y Ejecutar con Docker
```bash
docker build -t cats-vs-dogs .
docker run -p 5000:5000 cats-vs-dogs
```

## 📄 Estructura del Proyecto
```
cats-vs-dogs/
│── app/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preparation.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── api.py
│   ├── train.py
│── dataset/                 # 📁 NO SE INCLUYE EN GIT (Imágenes de entrenamiento)
│── models/                  # 📁 NO SE INCLUYE EN GIT (Modelo entrenado)
│── archive.zip              # 📁 NO SE INCLUYE EN GIT (Descargar de aquí: https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/data)
│── requirements.txt         # 📄 Dependencias
│── Dockerfile               # 📄 Configuración para Docker
│── main.py                  # 📄 Punto de entrada
│── README.md                # 📄 Documentación
│── .gitignore               # 📄 Archivos ignorados en Git
```

## 📊 Visualización de Resultados
El entrenamiento genera gráficas de **precisión** y **pérdida** a lo largo de las épocas usando `matplotlib`.

## 📦 Instalación de Dependencias
Si no usas Docker, instala las dependencias manualmente:
```bash
pip install -r requirements.txt
```

## 🖼 Subir Imágenes para Clasificación
Para predecir si una imagen es de un **perro o un gato**, usa la API:
```bash
curl -X POST -F "file=@imagen.jpg" http://localhost:5000/predict
```
Ejemplo de respuesta JSON:
```json
{
    "filename": "imagen.jpg",
    "prediction": "🐱 Es un GATO"
}
```

## 📜 Licencia
Este proyecto es de código abierto. Puedes usarlo y modificarlo libremente. 🚀🔥

