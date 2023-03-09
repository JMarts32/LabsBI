import numpy as np
import pandas as pd
import joblib

# Proceso de prueba del cliente
filename = 'PolinomialRegression.joblib' # Ubicación del archivo entregado
df_recent = pd.read_csv('../data/data_test.csv') # Lectura de los datos recientes

# Lee el archivo y carga el modelo
pipeline = load(filename)

y_true = pd.read_csv('MotorAlpes_validation.csv') # La columna que solo el cliente tiene
y_predicted =  pipeline.predict(df_recent)

# Calcula el desempeño del modelo
np.sqrt(mse(y_true, y_predicted))
