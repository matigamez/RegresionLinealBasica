# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos
data = pd.read_csv('./data/house-prices.csv')

# Exploración de datos
print(data.head())  
print(data.info())  # Información general, para detectar valores nulos y tipos de datos
print(data.describe())  # Estadísticas descriptivas, para detectar valores atípicos

# Verificar si existen valores nulos
print(data.isnull().sum())

# Seleccionar las columnas relevantes para el modelo
features = data[['SqFt', 'Bedrooms', 'Bathrooms']]
target = data['Price']

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Entrenamiento del modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio  y R-cuadrado 
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Error cuadrático medio (MSE): {mse}")
print(f"R-cuadrado (R²): {r2}")

# Predicción: Usar el modelo para predecir el precio de una casa con características dadas
new_house = pd.DataFrame([[2000, 3, 2]], columns=['SqFt', 'Bedrooms', 'Bathrooms'])  # Crear DataFrame con las columnas correctas
predicted_price = model.predict(new_house)
print(f"Precio de venta estimado para la nueva casa: {predicted_price[0]}")
