#!/bin/bash

echo "🚀 Iniciando build en Railway..."

# Instalar dependencias
echo "📦 Instalando dependencias..."
pip install -r requirements.txt

# Verificar que el archivo de datos existe
if [ ! -f "wheaterPba3Completo.csv" ]; then
    echo "❌ Error: No se encontró wheaterPba3Completo.csv"
    exit 1
fi

# Entrenar modelos
echo "🤖 Entrenando modelos..."
python setup_models.py

# Verificar que se generaron los modelos
echo "✅ Verificando modelos generados..."
ls -la *.pkl

echo "🎉 Build completado exitosamente!" 