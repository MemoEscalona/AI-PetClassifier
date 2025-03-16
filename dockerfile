# Usa una imagen base de Python con TensorFlow
FROM tensorflow/tensorflow:latest

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar archivos del proyecto al contenedor
COPY requirements.txt requirements.txt
COPY main.py main.py
COPY app/ app/
COPY models/ models/
COPY dataset/ dataset/
COPY static/ static/
COPY archive.zip archive.zip

# Instalar dependencias
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Exponer el puerto en el que Flask correrá
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "main.py", "--serve"]