#!/usr/bin/env python3
"""
Script para entrenar todos los modelos automáticamente.
Se ejecuta en Render.com para generar los archivos .pkl necesarios.
"""

import os
import sys
import gc

def main():
    print("Iniciando entrenamiento de modelos...")
    
    # Verificar si existe el archivo de datos
    if not os.path.exists('wheaterPba3Completo.csv'):
        print("Error: No se encontró wheaterPba3Completo.csv")
        print("Los modelos no se pueden entrenar sin datos.")
        return
    
    # Lista de scripts de entrenamiento en orden de prioridad
    training_scripts = [
        'entrenamiento_clustering.py',
        'entrenamiento_confort.py',
        'entrenamiento_lluvia.py',
        'entrenamiento_evaporacion.py',
        'entrenamiento_asociacion.py'
    ]
    
    # Ejecutar cada script de entrenamiento
    for script in training_scripts:
        if os.path.exists(script):
            print(f"\n{'='*50}")
            print(f"Entrenando modelo: {script}")
            print(f"{'='*50}")
            try:
                # Importar y ejecutar el script
                exec(open(script).read())
                print(f"✅ {script} completado exitosamente")
                
                # Limpiar memoria después de cada script
                gc.collect()
                
            except Exception as e:
                print(f"❌ Error en {script}: {e}")
                print("Continuando con el siguiente script...")
        else:
            print(f"⚠️ Script no encontrado: {script}")
    
    print(f"\n{'='*50}")
    print("Entrenamiento de modelos completado!")
    print(f"{'='*50}")
    
    # Verificar archivos generados
    expected_files = [
        'modelo_clustering.pkl',
        'modelo_confort.pkl', 
        'modelo_lluvia.pkl',
        'modelo_evaporacion.pkl',
        'modelo_asociacion.pkl',
        'scaler_clustering.pkl'
    ]
    
    print("\nVerificando archivos generados:")
    total_size = 0
    for file in expected_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)  # MB
            total_size += size
            print(f"✅ {file} ({size:.1f} MB)")
        else:
            print(f"❌ {file} - NO ENCONTRADO")
    
    print(f"\nTamaño total de modelos: {total_size:.1f} MB")
    
    if total_size > 0:
        print("🎉 ¡Entrenamiento exitoso! Los modelos están listos para usar.")
    else:
        print("⚠️ No se generaron modelos. Revisa los errores arriba.")

if __name__ == "__main__":
    main() 