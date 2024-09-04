# Usa una imagen base oficial de Python 3.12.5
FROM python:3.12.5-slim

RUN mkdir /video-analysis-with-gpt-4o

# Establece el directorio de trabajo en /app
WORKDIR /video-analysis-with-gpt-4o

# Copia los archivos de requisitos a la imagen
COPY requirements.txt /video-analysis-with-gpt-4o/requirements.txt

# Instala las dependencias
RUN pip install --no-cache-dir -r /video-analysis-with-gpt-4o/requirements.txt

# Copia el resto de los archivos de la aplicación
COPY . .

# Expone el puerto en el que la aplicación correrá
EXPOSE 8000

# Define el comando por defecto para ejecutar la aplicación
CMD ["streamlit", "run", "video-analysis-with-gpt-4o.py"]
