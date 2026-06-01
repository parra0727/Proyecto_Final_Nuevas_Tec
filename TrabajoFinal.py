import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────
# 1. CARGAR EL DATASET
# ─────────────────────────────────────────
ruta = 'dataset_banco.csv'  # Cambia esto si el archivo está en otra carpeta
data = pd.read_csv(ruta)

print("=== DATASET ORIGINAL ===")
print("Forma:", data.shape)
print(data.head())
data.info()

# ─────────────────────────────────────────
# 2.1 DATOS FALTANTES
# ─────────────────────────────────────────
print("\n=== VALORES NULOS POR COLUMNA ===")
print(data.isnull().sum())

data.dropna(inplace=True)

print("\nDespués de eliminar nulos:")
data.info()

# ─────────────────────────────────────────
# 2.2 COLUMNAS IRRELEVANTES
# ─────────────────────────────────────────
cols_cat = ['job', 'marital', 'education', 'default', 'housing',
            'loan', 'contact', 'month', 'poutcome', 'y']

print("\n=== SUBNIVELES POR COLUMNA CATEGÓRICA ===")
for col in cols_cat:
    print(f"  {col}: {data[col].nunique()} subnivel(es)")

print("\n=== ESTADÍSTICAS COLUMNAS NUMÉRICAS ===")
print(data.describe())
# Si std == 0 en alguna columna, esa columna tiene un solo valor y se eliminaría
# En este caso todas tienen variabilidad, no se elimina ninguna

# ─────────────────────────────────────────
# 2.3 FILAS REPETIDAS
# ─────────────────────────────────────────
print(f"\n=== DUPLICADOS ===")
print(f"Antes:  {data.shape}")
data.drop_duplicates(inplace=True)
print(f"Después: {data.shape}")

# ─────────────────────────────────────────
# 2.4 OUTLIERS EN VARIABLES NUMÉRICAS
# ─────────────────────────────────────────
cols_num = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

fig, ax = plt.subplots(nrows=7, ncols=1, figsize=(8, 30))
fig.subplots_adjust(hspace=0.5)
for i, col in enumerate(cols_num):
    sns.boxplot(x=col, data=data, ax=ax[i])
    ax[i].set_title(col)
plt.savefig('boxplots_outliers.png')
print("\nBoxplots guardados en 'boxplots_outliers.png'")

# age > 100 → imposible, error de captura
print(f"\nAntes de limpiar age:      {data.shape}")
data = data[data['age'] <= 100]
print(f"Después de limpiar age:    {data.shape}")

# duration <= 0 → una llamada no puede durar tiempo negativo
print(f"Antes de limpiar duration: {data.shape}")
data = data[data['duration'] > 0]
print(f"Después de limpiar duration: {data.shape}")

# previous > 100 → valor extremo, probablemente error
print(f"Antes de limpiar previous: {data.shape}")
data = data[data['previous'] <= 100]
print(f"Después de limpiar previous: {data.shape}")

# ─────────────────────────────────────────
# 2.5 ERRORES TIPOGRÁFICOS EN CATEGÓRICAS
# ─────────────────────────────────────────

# Paso 1: todo a minúsculas
for column in data.columns:
    if column in cols_cat:
        data[column] = data[column].str.lower()

# job: 'admin.' → 'administrative'
print("\njob antes:", data['job'].unique())
data['job'] = data['job'].str.replace('admin.', 'administrative', regex=False)
print("job después:", data['job'].unique())

# marital: 'div.' → 'divorced'
print("\nmarital antes:", data['marital'].unique())
data['marital'] = data['marital'].str.replace('div.', 'divorced', regex=False)
print("marital después:", data['marital'].unique())

# education: 'sec.' → 'secondary', 'unk' → 'unknown'
print("\neducation antes:", data['education'].unique())
data['education'] = data['education'].str.replace('sec.', 'secondary', regex=False)
data.loc[data['education'] == 'unk', 'education'] = 'unknown'
print("education después:", data['education'].unique())

# contact: 'phone' → 'telephone', 'mobile' → 'cellular'
print("\ncontact antes:", data['contact'].unique())
data.loc[data['contact'] == 'phone', 'contact'] = 'telephone'
data.loc[data['contact'] == 'mobile', 'contact'] = 'cellular'
print("contact después:", data['contact'].unique())

# poutcome: 'unk' → 'unknown'
print("\npoutcome antes:", data['poutcome'].unique())
data.loc[data['poutcome'] == 'unk', 'poutcome'] = 'unknown'
print("poutcome después:", data['poutcome'].unique())

# ─────────────────────────────────────────
# 3. RESULTADO FINAL
# ─────────────────────────────────────────
print("\n=== RESULTADO FINAL ===")
print(f"Dataset original:  45.215 filas x 17 columnas")
print(f"Dataset limpio:    {data.shape[0]} filas x {data.shape[1]} columnas")
print(f"Registros eliminados: {45215 - data.shape[0]}")

ruta_salida = 'dataset_banco_limpio.csv'
data.to_csv(ruta_salida, index=False)
print(f"\nDataset limpio guardado en: {ruta_salida}")