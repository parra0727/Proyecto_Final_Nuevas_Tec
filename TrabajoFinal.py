# =============================================================================
# PROYECTO FINAL - LIMPIEZA DE DATOS CON PANDAS
# Dataset: Bank Marketing (campañas de telemarketing de un banco portugués)
# Objetivo: Detectar y corregir problemas de calidad en los datos para dejar
#           el dataset listo para análisis o modelos de machine learning.
# =============================================================================

import pandas as pd          # Manipulación y análisis de datos tabulares
import matplotlib.pyplot as plt  # Generación de gráficas
import seaborn as sns            # Visualizaciones estadísticas sobre matplotlib

# ─────────────────────────────────────────
# 1. CARGAR EL DATASET
# ─────────────────────────────────────────
# Se carga el CSV con pandas. read_csv detecta automáticamente el separador
# y los tipos de datos de cada columna.
ruta = 'dataset_banco.csv'
data = pd.read_csv(ruta)

print("=== DATASET ORIGINAL ===")
print("Forma:", data.shape)   # (filas, columnas) → dimensión del dataset
print(data.head())             # Primeras 5 filas para inspección rápida
data.info()                    # Tipos de datos y conteo de no-nulos por columna

# ─────────────────────────────────────────
# 2.1 DATOS FALTANTES (valores nulos)
# ─────────────────────────────────────────
# isnull().sum() cuenta cuántos NaN hay en cada columna.
# Si hay nulos, los registros incompletos se eliminan con dropna()
# porque no aportan información confiable al análisis.
print("\n=== VALORES NULOS POR COLUMNA ===")
print(data.isnull().sum())

data.dropna(inplace=True)  # Elimina filas con al menos un valor nulo

print("\nDespués de eliminar nulos:")
data.info()

# ─────────────────────────────────────────
# 2.2 COLUMNAS IRRELEVANTES
# ─────────────────────────────────────────
# Se revisan las columnas categóricas para detectar categorías con un solo
# valor (std == 0 en categóricas equivale a columna constante → no aporta
# información y se podría eliminar).
cols_cat = ['job', 'marital', 'education', 'default', 'housing',
            'loan', 'contact', 'month', 'poutcome', 'y']

print("\n=== SUBNIVELES POR COLUMNA CATEGÓRICA ===")
for col in cols_cat:
    # nunique() devuelve el número de valores únicos; si es 1, la columna es constante
    print(f"  {col}: {data[col].nunique()} subnivel(es)")

print("\n=== ESTADÍSTICAS COLUMNAS NUMÉRICAS ===")
print(data.describe())
# Si std == 0 en alguna columna numérica, esa columna tiene un solo valor → se eliminaría.
# En este dataset todas las columnas numéricas tienen variabilidad, no se elimina ninguna.

# ─────────────────────────────────────────
# 2.3 FILAS REPETIDAS (duplicados)
# ─────────────────────────────────────────
# Un registro duplicado infla artificialmente ciertos grupos y sesga
# los análisis. drop_duplicates() conserva solo la primera aparición.
print(f"\n=== DUPLICADOS ===")
print(f"Antes:  {data.shape}")
data.drop_duplicates(inplace=True)
print(f"Después: {data.shape}")

# ─────────────────────────────────────────
# 2.4 OUTLIERS EN VARIABLES NUMÉRICAS
# ─────────────────────────────────────────
# Un outlier es un valor que se aleja demasiado del rango esperado;
# puede ser un error de captura o un caso extremo real.
# Se generan boxplots para visualizar la distribución de cada variable
# numérica e identificar visualmente los valores atípicos.
cols_num = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

fig, ax = plt.subplots(nrows=7, ncols=1, figsize=(8, 30))
fig.subplots_adjust(hspace=0.5)
for i, col in enumerate(cols_num):
    sns.boxplot(x=col, data=data, ax=ax[i])
    ax[i].set_title(col)
plt.savefig('boxplots_outliers.png')  # Se guarda la imagen para el informe
print("\nBoxplots guardados en 'boxplots_outliers.png'")

# Regla de negocio: age > 100 es imposible biológicamente → error de captura
print(f"\nAntes de limpiar age:      {data.shape}")
data = data[data['age'] <= 100]
print(f"Después de limpiar age:    {data.shape}")

# Regla de negocio: duration <= 0 es imposible (una llamada no puede durar 0 seg)
print(f"Antes de limpiar duration: {data.shape}")
data = data[data['duration'] > 0]
print(f"Después de limpiar duration: {data.shape}")

# Regla de negocio: previous > 100 contactos previos es un valor extremo irreal
print(f"Antes de limpiar previous: {data.shape}")
data = data[data['previous'] <= 100]
print(f"Después de limpiar previous: {data.shape}")

# ─────────────────────────────────────────
# 2.5 ERRORES TIPOGRÁFICOS EN CATEGÓRICAS
# ─────────────────────────────────────────
# Las columnas categóricas suelen tener inconsistencias de escritura:
# abreviaciones, mayúsculas mezcladas o sinónimos que representan
# el mismo concepto. Se estandarizan para que el análisis no trate
# categorías iguales como si fueran diferentes.

# Paso 1: pasar todo a minúsculas para eliminar diferencias por capitalización
for column in data.columns:
    if column in cols_cat:
        data[column] = data[column].str.lower()

# 'admin.' es abreviación de 'administrative' → se unifica el nombre
print("\njob antes:", data['job'].unique())
data['job'] = data['job'].str.replace('admin.', 'administrative', regex=False)
print("job después:", data['job'].unique())

# 'div.' es abreviación de 'divorced' → se escribe el nombre completo
print("\nmarital antes:", data['marital'].unique())
data['marital'] = data['marital'].str.replace('div.', 'divorced', regex=False)
print("marital después:", data['marital'].unique())

# 'sec.' → 'secondary' (nombre completo), 'unk' → 'unknown' (valor estándar)
print("\neducation antes:", data['education'].unique())
data['education'] = data['education'].str.replace('sec.', 'secondary', regex=False)
data.loc[data['education'] == 'unk', 'education'] = 'unknown'
print("education después:", data['education'].unique())

# 'phone' → 'telephone' y 'mobile' → 'cellular' para estandarizar con el resto
print("\ncontact antes:", data['contact'].unique())
data.loc[data['contact'] == 'phone', 'contact'] = 'telephone'
data.loc[data['contact'] == 'mobile', 'contact'] = 'cellular'
print("contact después:", data['contact'].unique())

# 'unk' → 'unknown' para mantener consistencia con el resto del dataset
print("\npoutcome antes:", data['poutcome'].unique())
data.loc[data['poutcome'] == 'unk', 'poutcome'] = 'unknown'
print("poutcome después:", data['poutcome'].unique())

# ─────────────────────────────────────────
# 3. RESULTADO FINAL
# ─────────────────────────────────────────
# Se exporta el dataset ya limpio a un nuevo CSV para no sobreescribir
# el original y mantener trazabilidad del proceso de limpieza.
print("\n=== RESULTADO FINAL ===")
print(f"Dataset original:  45.215 filas x 17 columnas")
print(f"Dataset limpio:    {data.shape[0]} filas x {data.shape[1]} columnas")
print(f"Registros eliminados: {45215 - data.shape[0]}")

ruta_salida = 'dataset_banco_limpio.csv'
data.to_csv(ruta_salida, index=False)  # index=False evita guardar el índice de pandas como columna extra
print(f"\nDataset limpio guardado en: {ruta_salida}")
