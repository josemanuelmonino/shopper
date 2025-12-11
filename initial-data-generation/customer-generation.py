import pandas as pd
import numpy as np

# Lista de 40 nombres españoles
nombres = [
    "Carlos Martínez", "Ana López", "Miguel Sánchez", "Laura González", "David Rodríguez",
    "Carmen Pérez", "Javier Gómez", "Elena Fernández", "Daniel Ruiz", "Sofía Díaz",
    "Pablo Hernández", "María García", "Álvaro Moreno", "Patricia Romero", "Sergio Álvarez",
    "Isabel Torres", "Francisco Navarro", "Raquel Jiménez", "Luis Ortega", "Clara Vargas",
    "Manuel Silva", "Natalia Castro", "Adrián Mendoza", "Beatriz Ríos", "Rubén Vega",
    "Cristina Flores", "Óscar Medina", "Silvia Guerrero", "Iván Reyes", "Virginia Campos",
    "Fernando Acosta", "Marta Herrera", "Raúl Peña", "Andrea Fuentes", "Juan Soto",
    "Lucía Mora", "Cristina García-Yáñez", "Lucía Pintos", "Miguel Conde", "José Moñino"
]

# Crear DataFrame
df = pd.DataFrame({
    'customer_id': [f"CUST-{i:05d}" for i in range(1, 41)],
    'name': nombres,
    'pass':"1234",
    'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F',
               'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F',
               'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F',
               'M', 'F', 'M', 'F', 'M', 'F', 'F', 'F', 'M', 'M']
})

# Generar email simple: nombre.apellido@gmail.com
def generar_email_simple(nombre):
    # Convertir a minúsculas y reemplazar espacios por puntos
    return nombre.lower().replace(' ', '.') + '@gmail.com'

df['email'] = df['name'].apply(generar_email_simple)

# Reordenar columnas
df = df[['customer_id', 'name', 'email', 'pass', 'gender']]

# Guardar como CSV
df.to_csv('df_clientes_final.csv', index=False, sep=';', encoding='utf-8')