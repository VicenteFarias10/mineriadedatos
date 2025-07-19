#!/usr/bin/env python3
"""
Script para entrenar todos los modelos automáticamente.
Se ejecuta en Render.com para generar los archivos .pkl necesarios.
"""

import os
import sys

def main():
    print("Iniciando entrenamiento de modelos...")
    
    # Verificar si existe el archivo de datos
    if not os.path.exists('wheaterPba3Completo.csv'):
        print("Error: No se encontró wheaterPba3Completo.csv")
        print("Los modelos no se pueden entrenar sin datos.")
        return
    
    # Lista de scripts de entrenamiento
    training_scripts = [
        'entrenamiento_clustering.py',
        'entrenamiento_asociacion.py', 
        'entrenamiento_confort.py',
        'entrenamiento_lluvia.py',
        'entrenamiento_evaporacion.py'
    ]
    
    # Ejecutar cada script de entrenamiento
    for script in training_scripts:
        if os.path.exists(script):
            print(f"\nEntrenando modelo: {script}")
            try:
                # Importar y ejecutar el script
                module_name = script.replace('.py', '')
                exec(open(script).read())
                print(f"✅ {script} completado exitosamente")
            except Exception as e:
                print(f"❌ Error en {script}: {e}")
        else:
            print(f"⚠️ Script no encontrado: {script}")
    
    print("\nEntrenamiento de modelos completado!")
    
    # Verificar archivos generados
    expected_files = [
        'modelo_clustering.pkl',
        'modelo_asociacion.pkl',
        'modelo_confort.pkl', 
        'modelo_lluvia.pkl',
        'modelo_evaporacion.pkl',
        'scaler_clustering.pkl'
    ]
    
    print("\nVerificando archivos generados:")
    for file in expected_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)  # MB
            print(f"✅ {file} ({size:.1f} MB)")
        else:
            print(f"❌ {file} - NO ENCONTRADO")

if __name__ == "__main__":
    main() 