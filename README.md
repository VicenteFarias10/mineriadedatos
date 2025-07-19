## se uso render.com en vez de github pages , ya que github pages no soporta el backend y tendria que haber convertido todo el python en javascript. por lo que se decidio usar render,tambien podria haber sido replit que tiene un funcionamiento similar, pero fue para darle una variacion, ademas, se tuvo que uasr git lfs debido a el alto tamaño de los archivos.

## Nota sobre archivos grandes

Los archivos de modelos (.pkl) son muy grandes (más de 100MB) y no se pueden subir a GitHub directamente. En su lugar:

1. Los archivos .pkl están excluidos del repositorio
2. Render.com entrenará automáticamente los modelos al desplegar
3. El script setup_models.py se ejecuta durante el build

# Minería de Datos - Análisis Climático

Aplicación web para analizar datos climáticos de Australia usando técnicas de machine learning.

## Qué hace esta aplicación

Esta aplicación te permite analizar datos climáticos de Australia de 5 maneras diferentes:

1. Clustering Climático - Agrupa zonas con climas similares
2. Análisis de Confort - Identifica zonas cómodas para vivir
3. Predicción de Lluvia - Predice si lloverá mañana
4. Análisis de Evaporación - Predice niveles de evaporación
5. Análisis de Asociación - Encuentra patrones climáticos

## Cómo funciona

1. Seleccionas una técnica de análisis
2. Subes un archivo CSV con datos climáticos
3. La aplicación procesa los datos y muestra resultados en mapas y gráficos

## Cómo usar en local

1. Instala las dependencias: `pip install -r requirements.txt`
2. Ejecuta los scripts de entrenamiento para generar los modelos
3. Ejecuta la aplicación: `python app.py`
4. Abre tu navegador en `http://localhost:3000`
5. Selecciona una técnica y sube el CSV

## Datos necesarios

el archivo CSV debe contener columnas como:
- Temperatura (MinTemp, MaxTemp)
- Humedad (Humidity9am, Humidity3pm)
- Lluvia (Rainfall)
- Ubicación (Latitud, Longitud)

## Despliegue

Esta aplicación está configurada para funcionar en Render.com
