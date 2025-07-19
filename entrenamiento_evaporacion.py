import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("Cargando datos...")
df = pd.read_csv('wheaterPba3Completo.csv')

print("Preparando datos para análisis de evaporación...")

# --- Mapeo de Localidades a Latitud y Longitud ---
location_coords = {
    'Albury': (-36.080556, 146.916389), 'BadgerysCreek': (-33.87, 150.73), 'Cobar': (-31.49, 145.84),
    'CoffsHarbour': (-30.30, 153.11), 'Moree': (-29.46, 149.85), 'Newcastle': (-32.93, 151.75),
    'NorahHead': (-33.28, 151.57), 'NorfolkIsland': (-29.04, 167.95), 'Penrith': (-33.75, 150.68),
    'Richmond': (-33.60, 150.75), 'Sydney': (-33.87, 151.21), 'SydneyAirport': (-33.94, 151.18),
    'WaggaWagga': (-35.12, 147.37), 'Williamtown': (-32.78, 151.84), 'Wollongong': (-34.42, 150.88),
    'Canberra': (-35.28, 149.13), 'Tuggeranong': (-35.43, 149.09), 'MountGinini': (-35.52, 148.78),
    'Ballarat': (-37.56, 143.85), 'Bendigo': (-36.76, 144.28), 'Beechworth': (-36.36, 146.68),
    'Brisbane': (-27.47, 153.02), 'Cairns': (-16.92, 145.77), 'GoldCoast': (-28.00, 153.43),
    'Townsville': (-19.26, 146.82), 'Adelaide': (-34.93, 138.60), 'Albany': (-35.02, 117.89),
    'Woomera': (-31.20, 136.82), 'Nuriootpa': (-34.46, 138.99), 'PearceRAAF': (-31.67, 116.01),
    'Perth': (-31.95, 115.86), 'PerthAirport': (-31.94, 115.97), 'SalmonGums': (-33.05, 121.64),
    'Walpole': (-34.97, 116.73), 'Hobart': (-42.88, 147.33), 'Launceston': (-41.43, 147.13),
    'Melbourne': (-37.81, 144.96), 'MelbourneAirport': (-37.67, 144.84), 'Mildura': (-34.20, 142.16),
    'Dartmoor': (-37.90, 141.28), 'Watsonia': (-37.72, 145.08), 'Portland': (-38.34, 141.60),
    'Nhil': (-36.33, 141.65), 'Uluru': (-25.35, 131.03), 'Darwin': (-12.46, 130.84),
    'Katherine': (-14.47, 132.27), 'AliceSprings': (-23.70, 133.88)
}

# Crear las columnas Latitud y Longitud si Location existe
if 'Location' in df.columns:
    df['Latitud'] = df['Location'].map(lambda x: location_coords.get(x, (np.nan, np.nan))[0])
    df['Longitud'] = df['Location'].map(lambda x: location_coords.get(x, (np.nan, np.nan))[1])
    print("Coordenadas mapeadas desde Location")
else:
    print("Columna Location no encontrada, usando Latitud y Longitud existentes")

# --- Seleccionar variables específicas (del código del usuario) ---
# Variables principales para evaporación
primary_vars = ['MaxTemp', 'Humidity3pm', 'Sunshine', 'Latitud', 'Longitud']

# Variables adicionales para análisis más completo
additional_vars = [
    'MinTemp', 'Humidity9am', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
    'Cloud9am', 'Cloud3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm',
    'Rainfall'
]

# Verificar qué columnas existen
available_primary = [col for col in primary_vars if col in df.columns]
available_additional = [col for col in additional_vars if col in df.columns]
all_available_vars = available_primary + available_additional

print(f"Variables principales disponibles: {available_primary}")
print(f"Variables adicionales disponibles: {available_additional}")

# Preparar datos (solo registros con Evaporation disponible)
print(f"Filas iniciales: {df.shape[0]}")
print(f"Valores faltantes en Evaporation: {df['Evaporation'].isnull().sum()}")

# Limpiar datos (siguiendo el enfoque del usuario)
df_evaporacion = df.dropna(subset=['Evaporation'] + available_primary)
print(f"Filas después de limpiar: {df_evaporacion.shape[0]}")

# Imputar variables adicionales con medianas si es necesario
for col in available_additional:
    if col in df_evaporacion.columns and df_evaporacion[col].isnull().any():
        median_val = df_evaporacion[col].median()
        df_evaporacion[col].fillna(median_val, inplace=True)
        print(f"NaNs en '{col}' imputados con la mediana: {median_val:.2f}")

print("\n--- Resumen de la Preparación de Datos ---")
print(f"Dimensiones del DataFrame final: {df_evaporacion.shape}")
print("\nConteo de valores faltantes después de la preparación:")
print(df_evaporacion[['Evaporation'] + all_available_vars].isnull().sum())
print("\nEstadísticas de Evaporation:")
print(df_evaporacion['Evaporation'].describe())

# --- Análisis Exploratorio de Datos ---
print("\n--- Análisis Exploratorio de Datos ---")

# 1. Distribución de Evaporation (del código del usuario)
plt.figure(figsize=(10, 6))
sns.histplot(df_evaporacion['Evaporation'], kde=True, bins=50)
plt.title('Distribución de la Evaporación Diaria (mm)', fontsize=16)
plt.xlabel('Evaporation (mm)', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('distribucion_evaporacion.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Scatter plots de variables principales (del código del usuario)
plt.figure(figsize=(18, 10))
for i, var in enumerate(available_primary):
    plt.subplot(2, 3, i + 1)
    sns.scatterplot(x=df_evaporacion[var], y=df_evaporacion['Evaporation'], alpha=0.6, s=10)
    plt.title(f'Evaporation vs. {var}', fontsize=14)
    plt.xlabel(var, fontsize=10)
    plt.ylabel('Evaporation (mm)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('scatter_evaporacion.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Matriz de Correlación (del código del usuario)
print("\nMatriz de Correlación entre Evaporation y variables predictoras:")
correlation_matrix = df_evaporacion[['Evaporation'] + available_primary].corr()
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlación', fontsize=16)
plt.savefig('matriz_correlacion_evaporacion.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Correlación con todas las variables
correlation_with_evaporation = df_evaporacion[all_available_vars + ['Evaporation']].corr()['Evaporation'].sort_values(ascending=False)

plt.figure(figsize=(10, 8))
correlation_with_evaporation.drop('Evaporation').plot(kind='barh')
plt.title('Correlación de Variables con Evaporation', fontsize=16)
plt.xlabel('Coeficiente de Correlación', fontsize=12)
plt.tight_layout()
plt.savefig('correlacion_evaporacion.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Preparación para el modelo ---
# Usar variables principales como en el código del usuario
features = available_primary
X = df_evaporacion[features]
y = df_evaporacion['Evaporation']

print(f"\nFeatures para el modelo (variables principales): {features}")
print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")

# Dividir en train y test (80-20 como en el código del usuario)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train set: {len(X_train)} registros")
print(f"Test set: {len(X_test)} registros")

# Escalar features
scaler_evaporacion = StandardScaler()
X_train_scaled = scaler_evaporacion.fit_transform(X_train)
X_test_scaled = scaler_evaporacion.transform(X_test)

# Convertir de nuevo a DataFrame para mantener los nombres de las columnas
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

# --- Entrenar múltiples modelos ---
print("\nEntrenando modelos de regresión...")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Lasso Regression': Lasso(alpha=0.1, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nEntrenando {name}...")
    model.fit(X_train_scaled_df, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test_scaled_df)
    
    # Métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    print(f"  R² Score: {r2:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE: {mae:.3f}")

# Seleccionar el mejor modelo
best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']
print(f"\nMejor modelo: {best_model_name} (R² = {results[best_model_name]['r2']:.3f})")

# --- Interpretación de coeficientes (del código del usuario) ---
if hasattr(best_model, 'coef_'):
    print("\n--- Interpretación de Coeficientes del Modelo ---")
    coefficients = pd.DataFrame({
        'Variable': X.columns, 
        'Coeficiente Estandarizado': best_model.coef_
    })
    print(coefficients.sort_values(by='Coeficiente Estandarizado', ascending=False))

# --- Guardar modelo ---
print("\nGuardando modelo de análisis de evaporación...")
modelo_completo_evaporacion = {
    'model': best_model,
    'scaler': scaler_evaporacion,
    'features': features,
    'best_model_name': best_model_name,
    'results': results,
    'correlation_with_evaporation': correlation_with_evaporation,
    'location_coords': location_coords
}

joblib.dump(modelo_completo_evaporacion, 'modelo_evaporacion.pkl')
print("Modelo de evaporación guardado como 'modelo_evaporacion.pkl'")

# --- Crear categorías de evaporación ---
def categorizar_evaporacion(valor):
    if valor < 3:
        return 'Baja'
    elif valor < 6:
        return 'Media'
    else:
        return 'Alta'

# Aplicar categorización a las predicciones del mejor modelo
y_pred_best = results[best_model_name]['y_pred']
y_pred_categorias = [categorizar_evaporacion(pred) for pred in y_pred_best]
y_test_categorias = [categorizar_evaporacion(real) for real in y_test]

# --- Visualizaciones de evaluación (del código del usuario) ---
plt.figure(figsize=(20, 12))

# 1. Comparación de modelos
plt.subplot(2, 4, 1)
model_names = list(results.keys())
r2_scores = [results[name]['r2'] for name in model_names]
plt.bar(model_names, r2_scores)
plt.title('Comparación de Modelos (R² Score)')
plt.ylabel('R² Score')
plt.xticks(rotation=45)

# 2. Predicciones vs Valores reales (del código del usuario)
plt.subplot(2, 4, 2)
plt.scatter(y_test, y_pred_best, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Línea de Predicción Ideal')
plt.xlabel('Evaporación Real (mm)')
plt.ylabel('Evaporación Predicha (mm)')
plt.title('Evaporación Real vs. Predicha')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 3. Distribución de errores (del código del usuario)
plt.subplot(2, 4, 3)
residuals = y_test - y_pred_best
sns.histplot(residuals, kde=True, bins=50, color='skyblue')
plt.title('Distribución de los Residuos del Modelo')
plt.xlabel('Residuos (Real - Predicho) (mm)')
plt.ylabel('Frecuencia')
plt.axvline(0, color='red', linestyle='--', label='Cero Residuo')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 4. Distribución de categorías predichas
plt.subplot(2, 4, 4)
categorias_pred = pd.Series(y_pred_categorias).value_counts()
plt.pie(categorias_pred.values, labels=categorias_pred.index, autopct='%1.1f%%')
plt.title('Distribución de Categorías Predichas')

# 5. Boxplot de evaporación por categoría
plt.subplot(2, 4, 5)
df_plot = pd.DataFrame({
    'Evaporacion': y_test,
    'Categoria': y_test_categorias
})
df_plot.boxplot(column='Evaporacion', by='Categoria', ax=plt.gca())
plt.title('Evaporación por Categoría')
plt.suptitle('')

# 6. Gráfico de residuos vs predicciones
plt.subplot(2, 4, 6)
plt.scatter(y_pred_best, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Evaporación Predicha')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos')

# 7. Importancia de features (si es Random Forest)
if hasattr(best_model, 'feature_importances_'):
    plt.subplot(2, 4, 7)
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Importancia')
    plt.title('Importancia de Features')
    plt.gca().invert_yaxis()

# 8. Coeficientes (si es modelo lineal)
elif hasattr(best_model, 'coef_'):
    plt.subplot(2, 4, 7)
    coef_df = pd.DataFrame({
        'feature': features,
        'coefficient': best_model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    plt.barh(range(len(coef_df)), coef_df['coefficient'])
    plt.yticks(range(len(coef_df)), coef_df['feature'])
    plt.xlabel('Coeficiente')
    plt.title('Coeficientes del Modelo')
    plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('analisis_evaporacion_completo.png', dpi=300, bbox_inches='tight')
print("Visualización completa guardada como 'analisis_evaporacion_completo.png'")
plt.close()

# Análisis adicional por ubicación
if 'Location' in df_evaporacion.columns:
    print("\nAnálisis por ubicación:")
    evaporacion_por_ubicacion = df_evaporacion.groupby('Location')['Evaporation'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    
    print("\nTop 10 ubicaciones con mayor evaporación promedio:")
    print(evaporacion_por_ubicacion.head(10))

print("\n--- Entrenamiento de análisis de evaporación completado ---")
print("Archivos generados:")
print("- modelo_evaporacion.pkl: Modelo entrenado")
print("- distribucion_evaporacion.png: Distribución de evaporación")
print("- scatter_evaporacion.png: Scatter plots de variables principales")
print("- matriz_correlacion_evaporacion.png: Matriz de correlación")
print("- correlacion_evaporacion.png: Correlación con todas las variables")
print("- analisis_evaporacion_completo.png: Análisis completo")
print(f"\nMejor modelo: {best_model_name}")
print(f"- R² Score: {results[best_model_name]['r2']:.3f}")
print(f"- RMSE: {results[best_model_name]['rmse']:.3f}")
print(f"- MAE: {results[best_model_name]['mae']:.3f}") 