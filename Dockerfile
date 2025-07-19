FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Entrenar modelos
RUN python setup_models.py

# Exponer puerto
EXPOSE 8080

# Comando de inicio
CMD gunicorn app:app --bind 0.0.0.0:8080 --workers 1 --timeout 120 