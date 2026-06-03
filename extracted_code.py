#pip install xgboost

#pip install prophet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
plt.style.use('ggplot')

df_2023 = pd.read_csv('2023.csv')
df_2024 = pd.read_csv('2024.csv')
df_2025 = pd.read_csv('2025.csv')

df_2023.info()

df_2024.info()

df_2025.info()

df_2023.duplicated().sum()

df_2024.duplicated().sum()

df_2025.duplicated().sum()

df_2023 = df_2023.drop_duplicates()
df_2024 = df_2024.drop_duplicates()
df_2025 = df_2025.drop_duplicates()

columnas_semanas = [col for col in df_2025.columns if col.startswith('2025')]

for col in columnas_semanas:

    mask = (
        df_2025[col]
        .astype(str)
        .str.contains(',', na=False)
    )

    if mask.any():

        print(f"\n{'='*50}")
        print(f"Columna: {col}")
        print(f"{'='*50}")

        ejemplos = df_2025.loc[
            mask,
            ['Channel', 'Material Description', 'Category', col]
        ].head(5)

        print(ejemplos)

        break

id_vars = [
    'Channel',
    'Material Description',
    'Category'
]

week_cols23 = [
    col for col in df_2023.columns
    if col not in id_vars
]

week_cols24 = [
    col for col in df_2024.columns
    if col not in id_vars
]

week_cols25 = [
    col for col in df_2025.columns
    if col not in id_vars
]

for col in week_cols23:
    df_2023[col] = pd.to_numeric(
        df_2023[col],
        errors='coerce'
    )

for col in week_cols24:
    df_2024[col] = pd.to_numeric(
        df_2024[col],
        errors='coerce'
    )

for col in week_cols25:
    df_2025[col] = df_2025[col].astype(str).str.replace(',', '', regex=False).str.strip()
    df_2025[col] = pd.to_numeric(df_2025[col], errors='coerce')


# Columnas clave
group_cols = [
    'Channel',
    'Material Description',
    'Category'
]

# Agrupar y sumar semanas
df_2023 = (
    df_2023
    .groupby(group_cols, as_index=False)
    .sum()
)

# Agrupar y sumar semanas
df_2024 = (
    df_2024
    .groupby(group_cols, as_index=False)
    .sum()
)

# Agrupar y sumar semanas
df_2025 = (
    df_2025
    .groupby(group_cols, as_index=False)
    .sum()
)

id_vars = ['Channel', 'Material Description', 'Category']

df_2023_long = df_2023.melt(
    id_vars=id_vars,
    var_name='Semana',
    value_name='Valor'
)

df_2024_long = df_2024.melt(
    id_vars=id_vars,
    var_name='Semana',
    value_name='Valor'
)

df_2025_long = df_2025.melt(
    id_vars=id_vars,
    var_name='Semana',
    value_name='Valor'
)

df_total = pd.concat(
    [df_2023_long, df_2024_long, df_2025_long],
    ignore_index=True
)

df_total.isnull().sum()

df_total['Año'] = df_total['Semana'].astype(str).str[:4].astype(int)

df_total['Semana'] = (df_total['Semana'].astype(str).str[-2:].astype(int))

df_total['Valor'] = df_total['Valor'].astype(int)

df_total['Material Description'].str.count(',').value_counts().sort_index()

for n in sorted(df_total['Material Description'].str.count(',').unique()):
    print(f"\n=== {n} comas ===")
    
    ejemplos = (
        df_total[
            df_total['Material Description'].str.count(',') == n
        ]['Material Description']
        .drop_duplicates()
        .head(5)
    )
    
    print(ejemplos.to_list())

df_total['TipoProducto'] = (
    df_total['Material Description']
    .str.split(',')
    .str[0]
    .str.strip()
)

# Crear fecha usando Año + Semana ISO
# Se usa el lunes de cada semana como referencia
df_total['Fecha'] = pd.to_datetime(
    df_total['Año'].astype(str) +
    df_total['Semana'].astype(str).str.zfill(2) +
    '1',
    format='%G%V%u'
)

df_total['Mes'] = df_total['Fecha'].dt.month


df_total = df_total[~((df_total['Año'] == 2025) & (df_total['Semana'] >= 34))].reset_index(drop=True)

print(df_total.shape)

df_total.describe()

print("\n")
print("Cuartiles:")

print(df_total['Valor'].quantile([0.25, 0.5, 0.75]))

print("\n")
print("Percentiles:")

print(df_total['Valor'].quantile([0.10, 0.90, 0.95, 0.99]))


print("\n")
print("Top 10 valores mas frecuentes:")

print(df_total['Valor'].value_counts().head(10))

print("\n")
print("Analisis por categoria:")

categoria_stats = (
    df_total
    .groupby('Category')['Valor']
    .agg([
        'count',
        'mean',
        'median',
        'std',
        'min',
        'max',
        'sum'
    ])
)

print(categoria_stats)

print("\n")
print("Analisis por año:")

anio_stats = (
    df_total
    .groupby('Año')['Valor']
    .agg([
        'count',
        'mean',
        'median',
        'std',
        'min',
        'max',
        'sum'
    ])
)

print(anio_stats)

print("\n")
print("Top 15 tipos de productos:")

print(
    df_total['TipoProducto']
    .value_counts()
    .head(15)
)


# ROTACIÓN DE INVENTARIO GENERAL Y POR PUNTO ESPECÍFICO (CANAL)
inventario = (
    df_total[df_total['Category'] == 'Channel Inv.']
    .groupby('Fecha')['Valor']
    .sum()
)

ventas = (
    df_total[df_total['Category'] == 'Sell-in']
    .groupby('Fecha')['Valor']
    .sum()
)

rotacion = (ventas / inventario).reset_index()
rotacion.columns = ['Fecha', 'Rotacion']

# Rotación por punto específico (Canal - Top 5 con más volumen)
top_clientes_nombres = (
    df_total
    .groupby('Channel')['Valor']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)

df_top_channels = df_total[df_total['Channel'].isin(top_clientes_nombres)]

inventario_ch = (
    df_top_channels[df_top_channels['Category'] == 'Channel Inv.']
    .groupby(['Fecha', 'Channel'])['Valor']
    .sum()
)

ventas_ch = (
    df_top_channels[df_top_channels['Category'] == 'Sell-in']
    .groupby(['Fecha', 'Channel'])['Valor']
    .sum()
)

rotacion_ch = (ventas_ch / inventario_ch).reset_index()
rotacion_ch.columns = ['Fecha', 'Channel', 'Rotacion']


plt.figure(figsize=(16, 7))

# Graficar rotación global
plt.plot(
    rotacion['Fecha'],
    rotacion['Rotacion'],
    label='Rotación Global (Promedio)',
    linewidth=3,
    color='black'
)

# Graficar rotación para el top de canales
for channel in top_clientes_nombres:
    ch_data = rotacion_ch[rotacion_ch['Channel'] == channel]
    plt.plot(
        ch_data['Fecha'],
        ch_data['Rotacion'],
        label=f'Rotación {channel}',
        alpha=0.7,
        linestyle='--'
)

plt.title('Rotación de Inventarios: General vs Puntos Específicos (Ventas / Inventario)')
plt.xlabel('Fecha')
plt.ylabel('Rotación (Ventas / Inventario)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


top_clientes = (
    df_total
    .groupby('Channel')['Valor']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(14,6))

top_clientes.plot(kind='bar')

plt.title('Top 10 Clientes')

plt.ylabel('Valor Total')

plt.xticks(rotation=45)

plt.show()

ventas_categoria = (
    df_total
    .groupby(
        ['Fecha', 'Category']
    )['Valor']
    .sum()
    .reset_index()
)

plt.figure(figsize=(16,6))

sns.lineplot(
    data=ventas_categoria,
    x='Fecha',
    y='Valor',
    hue='Category'
)

plt.title('Comportamiento Temporal por Categoria')

plt.xticks(rotation=45)

plt.show()

ventas_mensuales = (
    df_total
    .groupby('Fecha')['Valor']
    .sum()
    .reset_index()
)

plt.figure(figsize=(16,6))

plt.plot(
    ventas_mensuales['Fecha'],
    ventas_mensuales['Valor']
)

plt.title('Comportamiento Temporal General')

plt.xlabel('Fecha')

plt.ylabel('Valor Total')

plt.xticks(rotation=45)

plt.show()

plt.figure(figsize=(12,6))

plt.hist(
    df_total['Valor'],
    bins=50
)

plt.title('Distribucion de Valor')

plt.xlabel('Valor')

plt.ylabel('Frecuencia')

plt.show()

plt.figure(figsize=(12,6))

sns.boxplot(
    x=df_total['Valor']
)

plt.title('Boxplot de Valor')

plt.show()

plt.figure(figsize=(12,6))

sns.boxplot(
    data=df_total,
    x='Category',
    y='Valor'
)

plt.title('Valor por Categoria')

plt.xticks(rotation=45)

plt.show()


ventas_semana = (
    df_total
    .groupby('Semana')['Valor']
    .sum()
)

plt.figure(figsize=(14,6))

plt.plot(
    ventas_semana.index,
    ventas_semana.values
)

plt.title('Comportamiento Semanal')

plt.xlabel('Semana')

plt.ylabel('Valor Total')

plt.show()

ventas_anio_semana = (
    df_total
    .groupby(['Año', 'Semana'])['Valor']
    .sum()
    .reset_index()
)

plt.figure(figsize=(16,6))

sns.lineplot(
    data=ventas_anio_semana,
    x='Semana',
    y='Valor',
    hue='Año'
)

plt.title('Comportamiento Semanal por Año')

plt.show()

top_productos = (
    df_total
    .groupby('TipoProducto')['Valor']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,6))

top_productos.plot(kind='bar')

plt.title('Top 10 Tipos de Productos')

plt.ylabel('Ventas Totales')

plt.xticks(rotation=45)

plt.show()

df_total.info()

corr = df_total[
    ['Año', 'Semana', 'Valor',]
].corr()

print("\n")
print("Matriz de correlacion:")

print(corr)

plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm'
)

plt.title('Matriz de Correlacion')

plt.show()

# ============================================================
# 1. FILTRAR Y AGREGAR SOLO VENTAS AL CONSUMIDOR (Cust. Sales)
#    Es la métrica más relevante para predecir demanda real
#    Agregamos por cliente y tipo de producto para evitar sesgos
# ============================================================
df_ventas = df_total[df_total['Category'] == 'Cust. Sales'].copy()
df_ventas = df_ventas.groupby(['Channel', 'TipoProducto', 'Año', 'Semana', 'Fecha', 'Mes'], as_index=False)['Valor'].sum()

# ============================================================
# 2. FEATURES DE TIEMPO (el modelo necesita entender temporalidad)
# ============================================================
# Semana del año como ángulo → captura estacionalidad cíclica
# Ejemplo: semana 1 y semana 52 son "cercanas" en el año
df_ventas['semana_sin'] = np.sin(2 * np.pi * df_ventas['Semana'] / 52)
df_ventas['semana_cos'] = np.cos(2 * np.pi * df_ventas['Semana'] / 52)

# ============================================================
# 3. FEATURES DE HISTORIAL (lag features)
#    Le decimos al modelo: "la semana pasada se vendió X"
# ============================================================
df_ventas = df_ventas.sort_values(['Channel', 'TipoProducto', 'Año', 'Semana'])

# Ventas de semanas anteriores por cliente+producto
df_ventas['lag_1'] = df_ventas.groupby(['Channel', 'TipoProducto'])['Valor'].shift(1)
df_ventas['lag_2'] = df_ventas.groupby(['Channel', 'TipoProducto'])['Valor'].shift(2)
df_ventas['lag_4'] = df_ventas.groupby(['Channel', 'TipoProducto'])['Valor'].shift(4)

# Promedio móvil de las últimas 4 semanas
df_ventas['media_movil_4'] = (
    df_ventas
    .groupby(['Channel', 'TipoProducto'])['Valor']
    .transform(lambda x: x.shift(1).rolling(4).mean())
)

# ============================================================
# 4. CODIFICAR VARIABLES CATEGÓRICAS
#    XGBoost solo entiende números
# ============================================================
le_channel      = LabelEncoder()
le_tipoproducto = LabelEncoder()

df_ventas['Channel_enc']      = le_channel.fit_transform(df_ventas['Channel'])
df_ventas['TipoProducto_enc'] = le_tipoproducto.fit_transform(df_ventas['TipoProducto'])

# ============================================================
# 5. ELIMINAR FILAS CON NaN (generados por los lags)
# ============================================================
df_ventas = df_ventas.dropna(subset=['lag_1', 'lag_2', 'lag_4', 'media_movil_4'])

print(df_ventas.shape)
print(df_ventas[['Channel', 'TipoProducto', 'Año', 'Semana', 'lag_1', 'media_movil_4', 'Valor']].head(10))


# ============================================================
# FEATURES Y TARGET
# ============================================================
features = [
    'Channel_enc',
    'TipoProducto_enc',
    'Año',
    'Semana',
    'semana_sin',
    'semana_cos',
    'lag_1',
    'lag_2',
    'lag_4',
    'media_movil_4'
]

target = 'Valor'

# ============================================================
# SPLIT TEMPORAL — NO aleatorio
#    Entrenamos con 2023-2024, predecimos 2025
#    En series de tiempo NO se hace train_test_split aleatorio
#    porque estarías usando el futuro para predecir el pasado
# ============================================================
train = df_ventas[df_ventas['Año'] < 2025]
test  = df_ventas[df_ventas['Año'] == 2025]

X_train = train[features]
y_train = train[target]

X_test  = test[features]
y_test  = test[target]

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ============================================================
# ENTRENAR MODELO
# ============================================================
modelo_xgb = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

modelo_xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# ============================================================
# EVALUACIÓN
# ============================================================
y_pred = modelo_xgb.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nMAE  (error promedio en unidades): {mae:.2f}")
print(f"RMSE (penaliza errores grandes):   {rmse:.2f}")
print(f"R²   (qué tanto explica el modelo): {r2:.4f}")

# ============================================================
# IMPORTANCIA DE FEATURES
# ============================================================
importancias = pd.Series(
    modelo_xgb.feature_importances_,
    index=features
).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
importancias.plot(kind='bar')
plt.title('Importancia de Variables - XGBoost')
plt.tight_layout()
plt.show()

# ============================================================
# DEFINIR CHURN
#    Un cliente-producto está en riesgo si tuvo 4 semanas
#    consecutivas con Valor = 0 en sus registros más recientes
# ============================================================
ultimas_semanas = (
    df_ventas
    .sort_values(['Channel', 'TipoProducto', 'Año', 'Semana'])
    .groupby(['Channel', 'TipoProducto'])
    .tail(4)  # últimas 4 semanas de cada combinación
)

churn_flags = (
    ultimas_semanas
    .groupby(['Channel', 'TipoProducto'])['Valor']
    .apply(lambda x: int((x == 0).all() and len(x) == 4))  # 1 si todas son 0 y tenemos las 4 semanas
    .reset_index()
    .rename(columns={'Valor': 'churn'})
)

print(f"Clientes en riesgo: {churn_flags['churn'].sum()}")
print(f"Clientes activos:   {(churn_flags['churn'] == 0).sum()}")

# ============================================================
# PREPARAR DATOS PARA CLASIFICACIÓN
# ============================================================
df_churn = df_ventas.merge(churn_flags, on=['Channel', 'TipoProducto'])

X_churn = df_churn[features]
y_churn = df_churn['churn']

# Split temporal
X_c_train = X_churn[df_churn['Año'] < 2025]
y_c_train = y_churn[df_churn['Año'] < 2025]
X_c_test  = X_churn[df_churn['Año'] == 2025]
y_c_test  = y_churn[df_churn['Año'] == 2025]

# ============================================================
# ENTRENAR CLASIFICADOR
# ============================================================
modelo_churn = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    scale_pos_weight=len(y_c_train[y_c_train==0]) / len(y_c_train[y_c_train==1]),
    random_state=42,
    n_jobs=-1
)

modelo_churn.fit(X_c_train, y_c_train)

# ============================================================
# EVALUACIÓN
# ============================================================
y_churn_pred = modelo_churn.predict(X_c_test)

print("\nReporte de clasificación:")
print(classification_report(y_c_test, y_churn_pred))

# ============================================================
# CLIENTES EN RIESGO — resultado accionable para la dirección
# ============================================================
test_churn = df_churn[df_churn['Año'] == 2025].copy()
test_churn['churn_pred'] = y_churn_pred

clientes_riesgo = (
    test_churn[test_churn['churn_pred'] == 1]
    [['Channel', 'TipoProducto']]
    .drop_duplicates()
    .sort_values('Channel')
)

print(f"\nClientes en riesgo detectados: {len(clientes_riesgo)}")
print(clientes_riesgo.head(15))


# ============================================================
# PREPARAR SERIE DE TIEMPO AGREGADA
#    Prophet necesita columnas 'ds' (fecha) y 'y' (valor)
# ============================================================
df_prophet = (
    df_total[df_total['Category'] == 'Cust. Sales']
    .groupby('Fecha')['Valor']
    .sum()
    .reset_index()
    .rename(columns={'Fecha': 'ds', 'Valor': 'y'})
)

# ============================================================
# ENTRENAR PROPHET
# ============================================================
modelo_prophet = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)

modelo_prophet.fit(df_prophet)

# ============================================================
# PREDECIR LAS PRÓXIMAS 12 SEMANAS
# ============================================================
futuro     = modelo_prophet.make_future_dataframe(periods=12, freq='W')
prediccion = modelo_prophet.predict(futuro)

# Gráfico de tendencia
modelo_prophet.plot(prediccion)
plt.title('Predicción de Ventas - Próximas 12 semanas')
plt.show()

# Componentes (tendencia + estacionalidad)
modelo_prophet.plot_components(prediccion)
plt.show()

# Ver predicciones futuras
print(prediccion[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))

df_forecast = df_total[
    (df_total['Category'] == 'Cust. Sales') &
    (df_total['Valor'] >= 0)
].copy()

df_forecast = (
    df_forecast
    .groupby(
        ['Channel', 'TipoProducto', 'Año', 'Semana', 'Fecha', 'Mes'],
        as_index=False
    )['Valor']
    .sum()
)

df_forecast['semana_sin'] = np.sin(
    2 * np.pi * df_forecast['Semana'] / 52
)

df_forecast['semana_cos'] = np.cos(
    2 * np.pi * df_forecast['Semana'] / 52
)

df_forecast = df_forecast.sort_values(
    ['Channel', 'TipoProducto', 'Año', 'Semana']
)

# ============================================================
# BLOQUE 4: FEATURES HISTÓRICAS (LAGS)
# Objetivo:
# Permitir que el modelo utilice ventas pasadas para
# predecir ventas futuras.
# ============================================================

grupo = df_forecast.groupby(
    ['Channel', 'TipoProducto']
)['Valor']

df_forecast['lag_1'] = grupo.shift(1)
df_forecast['lag_2'] = grupo.shift(2)
df_forecast['lag_4'] = grupo.shift(4)
df_forecast['lag_8'] = grupo.shift(8)
df_forecast['lag_12'] = grupo.shift(12)

# ============================================================
# BLOQUE 5: TENDENCIA DE CORTO Y MEDIANO PLAZO
# Objetivo:
# Capturar el comportamiento reciente de ventas.
# ============================================================

df_forecast['media_movil_4'] = (
    grupo.transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    )
)

df_forecast['media_movil_8'] = (
    grupo.transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean()
    )
)

df_forecast['media_movil_12'] = (
    grupo.transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).mean()
    )
)

# ============================================================
# BLOQUE 6: MEDIDAS DE VARIABILIDAD
# Objetivo:
# Indicar si el comportamiento histórico es estable
# o altamente variable.
# ============================================================

df_forecast['std_4'] = (
    grupo.transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).std()
    )
)

df_forecast['std_8'] = (
    grupo.transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).std()
    )
)

df_forecast[['std_4', 'std_8']] = (
    df_forecast[['std_4', 'std_8']]
    .fillna(0)
)

# ============================================================
# BLOQUE 7: CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# Objetivo:
# Transformar variables categóricas a formato numérico.
# ============================================================

from sklearn.preprocessing import LabelEncoder

le_channel = LabelEncoder()
le_producto = LabelEncoder()

df_forecast['Channel_enc'] = (
    le_channel.fit_transform(df_forecast['Channel'])
)

df_forecast['TipoProducto_enc'] = (
    le_producto.fit_transform(df_forecast['TipoProducto'])
)

# ============================================================
# BLOQUE 8: LIMPIEZA FINAL
# Objetivo:
# Eliminar registros sin suficiente historial.
# ============================================================

lag_cols = [
    'lag_1',
    'lag_2',
    'lag_4',
    'lag_8',
    'lag_12',
    'media_movil_4',
    'media_movil_8',
    'media_movil_12',
]

df_forecast = (
    df_forecast
    .dropna(subset=lag_cols)
    .reset_index(drop=True)
)

# ============================================================
# BLOQUE 9: DEFINICIÓN DE VARIABLES
# Objetivo:
# Definir predictores y variable objetivo.
# ============================================================

features = [
    'Channel_enc',
    'TipoProducto_enc',
    'Año',
    'Semana',
    'Mes',
    'semana_sin',
    'semana_cos',
    'lag_1',
    'lag_2',
    'lag_4',
    'lag_8',
    'lag_12',
    'media_movil_4',
    'media_movil_8',
    'media_movil_12',
    'std_4',
    'std_8'
]

target = 'Valor'

# ============================================================
# BLOQUE 10: PARTICIÓN TEMPORAL
# Objetivo:
# Separar entrenamiento, validación y prueba respetando
# la secuencia temporal.
# ============================================================

mask_val = (
    (df_forecast['Año'] == 2024) &
    (df_forecast['Semana'] > 40)
)

train = df_forecast[
    (~mask_val) &
    (df_forecast['Año'] < 2025)
]

validacion = df_forecast[
    mask_val
]

test = df_forecast[
    df_forecast['Año'] == 2025
]

X_train = train[features]
y_train = train[target]

X_val = validacion[features]
y_val = validacion[target]

X_test = test[features]
y_test = test[target]

# ============================================================
# BLOQUE 11: ENTRENAMIENTO DEL MODELO
# Objetivo:
# Ajustar un modelo XGBoost optimizado para forecasting.
# ============================================================

import xgboost as xgb

modelo_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)

modelo_xgb.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# ============================================================
# BLOQUE 12: PREDICCIONES
# Objetivo:
# Generar predicciones sobre el conjunto de prueba.
# ============================================================

y_pred = modelo_xgb.predict(X_test)

# ============================================================
# BLOQUE 13: EVALUACIÓN DEL MODELO
# Objetivo:
# Medir precisión y capacidad predictiva.
# ============================================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mask_nonzero = y_test != 0

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(
    y_test,
    y_pred
)

mape = (
    np.mean(
        np.abs(
            (y_test[mask_nonzero] - y_pred[mask_nonzero])
            / y_test[mask_nonzero]
        )
    ) * 100
)

smape = (
    np.mean(
        2 * np.abs(y_pred - y_test)
        /
        (
            np.abs(y_test)
            + np.abs(y_pred)
            + 1e-8
        )
    ) * 100
)

pd.DataFrame({
    'MAE': [round(mae, 2)],
    'RMSE': [round(rmse, 2)],
    'R2': [round(r2, 4)],
    'MAPE_%': [round(mape, 2)],
    'SMAPE_%': [round(smape, 2)]
})

# ============================================================
# BLOQUE 14: IMPORTANCIA DE VARIABLES
# Objetivo:
# Identificar qué factores explican más las ventas.
# ============================================================

importancias = pd.DataFrame({
    'Variable': features,
    'Importancia': modelo_xgb.feature_importances_
})

importancias = (
    importancias
    .sort_values(
        'Importancia',
        ascending=False
    )
)

importancias.head(15)

# ============================================================
# BLOQUE 15: COMPARACIÓN REAL VS PREDICHO
# Objetivo:
# Analizar visualmente el comportamiento agregado.
# ============================================================

comparacion = pd.DataFrame({
    'Fecha': test['Fecha'],
    'Real': y_test,
    'Predicho': y_pred
})

comparacion = (
    comparacion
    .groupby('Fecha')
    .sum()
    .reset_index()
)

plt.figure(figsize=(14, 5))

plt.plot(
    comparacion['Fecha'],
    comparacion['Real'],
    label='Real'
)

plt.plot(
    comparacion['Fecha'],
    comparacion['Predicho'],
    '--',
    label='Predicho'
)

plt.legend()
plt.tight_layout()
plt.show()

y_test.describe()

(y_test == 0).mean() * 100

mask = y_test > 10

mae_alt = mean_absolute_error(
    y_test[mask],
    y_pred[mask]
)

rmse_alt = np.sqrt(
    mean_squared_error(
        y_test[mask],
        y_pred[mask]
    )
)

r2_alt = r2_score(
    y_test[mask],
    y_pred[mask]
)

print(mae_alt)
print(rmse_alt)
print(r2_alt)

# ============================================================
# PASO 1: FILTRAR SOLO VENTAS AL CONSUMIDOR
# ============================================================
df_ventas = df_total[df_total['Category'] == 'Cust. Sales'].copy()
df_ventas = df_ventas.groupby(
    ['Channel', 'TipoProducto', 'Año', 'Semana', 'Fecha', 'Mes'],
    as_index=False
)['Valor'].sum()

# ============================================================
# PASO 2: FEATURES DE TEMPORALIDAD CÍCLICA
# Semana como seno/coseno para que el modelo entienda
# que semana 52 y semana 1 son "cercanas" en el calendario
# ============================================================
df_ventas['semana_sin'] = np.sin(2 * np.pi * df_ventas['Semana'] / 52)
df_ventas['semana_cos'] = np.cos(2 * np.pi * df_ventas['Semana'] / 52)

# ============================================================
# PASO 3: LAG FEATURES
# Se eliminó lag_52 porque con solo 3 años de datos
# consume todo 2023 y deja el train sin filas.
# lag_1, lag_2, lag_4 son suficientes para capturar
# el comportamiento reciente de cada cliente-producto.
# ============================================================
df_ventas = df_ventas.sort_values(['Channel', 'TipoProducto', 'Año', 'Semana'])

grupo = df_ventas.groupby(['Channel', 'TipoProducto'])['Valor']

df_ventas['lag_1']         = grupo.shift(1)
df_ventas['lag_2']         = grupo.shift(2)
df_ventas['lag_4']         = grupo.shift(4)
df_ventas['media_movil_4'] = grupo.transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
df_ventas['media_movil_8'] = grupo.transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean())

# ============================================================
# PASO 4: CODIFICAR VARIABLES CATEGÓRICAS
# ============================================================
le_channel      = LabelEncoder()
le_tipoproducto = LabelEncoder()

df_ventas['Channel_enc']      = le_channel.fit_transform(df_ventas['Channel'])
df_ventas['TipoProducto_enc'] = le_tipoproducto.fit_transform(df_ventas['TipoProducto'])

# ============================================================
# PASO 5: ELIMINAR NaN DE LAGS
# ============================================================
lag_cols = ['lag_1', 'lag_2', 'lag_4', 'media_movil_4', 'media_movil_8']
df_ventas = df_ventas.dropna(subset=lag_cols).reset_index(drop=True)

print(f"Filas tras limpiar NaN: {df_ventas.shape[0]:,}")
print(df_ventas[['Channel', 'TipoProducto', 'Año', 'Semana', 'lag_1', 'media_movil_4', 'Valor']].head(10))

# ============================================================
# PASO 6: FEATURES Y TARGET
# ============================================================
features = [
    'Channel_enc',
    'TipoProducto_enc',
    'Año',
    'Semana',
    'Mes',
    'semana_sin',
    'semana_cos',
    'lag_1',
    'lag_2',
    'lag_4',
    'media_movil_4',
    'media_movil_8',
]

target = 'Valor'

# ============================================================
# PASO 7: SPLIT TEMPORAL CORREGIDO
#
# ANTES (error): train = Año <= 2023 → quedaba vacío porque
# lag_4 ya consume las primeras semanas de 2023.
#
# AHORA: train = 2023 + primeras 40 semanas de 2024
#         val  = últimas 12 semanas de 2024 (para early stopping)
#         test = 2025
#
# Esto garantiza que el modelo tenga suficientes datos
# para aprender y que la validación sea temporalmente posterior.
# ============================================================
mask_val = (df_ventas['Año'] == 2024) & (df_ventas['Semana'] > 40)

train      = df_ventas[~mask_val & (df_ventas['Año'] < 2025)]
validacion = df_ventas[mask_val]
test       = df_ventas[df_ventas['Año'] == 2025]

X_train = train[features];      y_train = train[target]
X_val   = validacion[features]; y_val   = validacion[target]
X_test  = test[features];       y_test  = test[target]

print(f"\nTrain: {X_train.shape} | Validación: {X_val.shape} | Test: {X_test.shape}")

# ============================================================
# PASO 8: ENTRENAR CON EARLY STOPPING
# early_stopping_rounds=30: si en 30 rondas el error de
# validación no mejora, se detiene el entrenamiento.
# Así encontramos el número óptimo de árboles automáticamente.
# ============================================================
modelo_xgb = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
)

modelo_xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

print(f"\nMejor número de árboles: {modelo_xgb.best_iteration}")

# ============================================================
# PASO 9: MÉTRICAS COMPLETAS
# ============================================================
y_pred = modelo_xgb.predict(X_test)

mask_nonzero = y_test != 0

mae   = mean_absolute_error(y_test, y_pred)
rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
r2    = r2_score(y_test, y_pred)
mape  = np.mean(np.abs((y_test[mask_nonzero] - y_pred[mask_nonzero]) / y_test[mask_nonzero])) * 100
smape = np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred) + 1e-8)) * 100

print("\n" + "="*55)
print("  MÉTRICAS DE EVALUACIÓN — XGBoost Regresor")
print("="*55)
print(f"  MAE   (error promedio en unidades) : {mae:.2f}")
print(f"  RMSE  (penaliza errores grandes)   : {rmse:.2f}")
print(f"  R²    (varianza explicada)         : {r2:.4f}  ({r2*100:.1f}%)")
print(f"  MAPE  (error % sobre no-cero)      : {mape:.2f}%")
print(f"  SMAPE (error % simétrico)          : {smape:.2f}%")
print("="*55)

if r2 >= 0.85:
    nivel = "EXCELENTE — el modelo captura muy bien la demanda"
elif r2 >= 0.70:
    nivel = "BUENO — predicciones útiles para decisiones operativas"
elif r2 >= 0.50:
    nivel = "MODERADO — sirve para tendencias, no para unidades exactas"
else:
    nivel = "BAJO — revisar features o estrategia de modelado"
print(f"  Interpretación R²: {nivel}")
print("="*55)

# ============================================================
# PASO 10: REAL vs PREDICHO
# ============================================================
comparacion = test[['Fecha']].copy().reset_index(drop=True)
comparacion['Real']     = y_test.values
comparacion['Predicho'] = y_pred

ventas_reales = comparacion.groupby('Fecha')['Real'].sum()
ventas_pred   = comparacion.groupby('Fecha')['Predicho'].sum()

plt.figure(figsize=(16, 5))
plt.plot(ventas_reales.index, ventas_reales.values, label='Real',     linewidth=2)
plt.plot(ventas_pred.index,   ventas_pred.values,   label='Predicho', linewidth=2, linestyle='--')
plt.title('Ventas Reales vs Predichas (2025) — XGBoost')
plt.xlabel('Semana')
plt.ylabel('Unidades')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================================================
# PASO 11: IMPORTANCIA DE VARIABLES
# ============================================================
importancias = pd.Series(
    modelo_xgb.feature_importances_,
    index=features
).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
importancias.plot(kind='barh')
plt.title('Importancia de Variables — XGBoost')
plt.xlabel('Importancia')
plt.tight_layout()
plt.show()

print("\nTop 5 variables más importantes:")
print(importancias.sort_values(ascending=False).head(5))

# ============================================================
# MODELO DE CHURN — VERSIÓN CORREGIDA
# ============================================================

# ============================================================
# PASO 1: DEFINIR CHURN CON DATOS DE 2024
# Un cliente-producto está en riesgo si sus últimas 4 semanas
# disponibles en 2024 tienen Valor = 0.
# Se usa 2024 (no todo el dataset) para evitar data leakage.
# ============================================================
df_2024 = df_ventas[df_ventas['Año'] == 2024].copy()

ultimas_4 = (
    df_2024
    .sort_values(['Channel', 'TipoProducto', 'Semana'])
    .groupby(['Channel', 'TipoProducto'])
    .tail(4)
)

churn_flags = (
    ultimas_4
    .groupby(['Channel', 'TipoProducto'])['Valor']
    .apply(lambda x: int((x == 0).all() and len(x) == 4))
    .reset_index()
    .rename(columns={'Valor': 'churn'})
)

n_riesgo  = churn_flags['churn'].sum()
n_activos = (churn_flags['churn'] == 0).sum()
print(f"Clientes-producto en riesgo : {n_riesgo}")
print(f"Clientes-producto activos   : {n_activos}")
print(f"Tasa de churn               : {n_riesgo / (n_riesgo + n_activos) * 100:.1f}%")

# ============================================================
# PASO 2: UNIR LABEL Y SPLIT TEMPORAL
# Train: 2023 + primeras 40 semanas de 2024
# Test : 2025
# El label (churn) viene de fin de 2024, así que el modelo
# de train aprende "¿qué perfil lleva a dejar de comprar?"
# ============================================================
df_churn = df_ventas.merge(churn_flags, on=['Channel', 'TipoProducto'], how='inner')

mask_val_c = (df_churn['Año'] == 2024) & (df_churn['Semana'] > 40)

train_c = df_churn[~mask_val_c & (df_churn['Año'] < 2025)]
test_c  = df_churn[df_churn['Año'] == 2025]

X_c_train = train_c[features]
y_c_train = train_c['churn']
X_c_test  = test_c[features]
y_c_test  = test_c['churn']

print(f"\nTrain churn: {X_c_train.shape} | Test churn: {X_c_test.shape}")
print(f"Distribución train: {y_c_train.value_counts().to_dict()}")
print(f"Distribución test : {y_c_test.value_counts().to_dict()}")

# ============================================================
# PASO 3: ENTRENAR CLASIFICADOR
# scale_pos_weight: si hay 9 activos por cada 1 en riesgo,
# le decimos al modelo que equivocarse en churn=1 pesa 9x más.
# Evita que el modelo diga "nadie hace churn" y tenga
# 90% de accuracy sin aprender nada real.
# ============================================================
n0 = (y_c_train == 0).sum()
n1 = (y_c_train == 1).sum()
ratio = n0 / max(n1, 1)

print(f"\nscale_pos_weight usado: {ratio:.2f}")

modelo_churn = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)

modelo_churn.fit(X_c_train, y_c_train)

# ============================================================
# PASO 4: MÉTRICAS DEL CLASIFICADOR
# Para churn, las métricas clave son:
#   Recall en "En riesgo": ¿cuántos clientes en riesgo detectamos?
#   Precision en "En riesgo": ¿cuántas alarmas son reales?
#   AUC-ROC: capacidad general de separar las dos clases
# ============================================================
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay
)

y_churn_pred  = modelo_churn.predict(X_c_test)
y_churn_proba = modelo_churn.predict_proba(X_c_test)[:, 1]

print("\n" + "="*55)
print("  MÉTRICAS DE EVALUACIÓN — XGBoost Churn")
print("="*55)
print(classification_report(
    y_c_test, y_churn_pred,
    target_names=['Activo', 'En riesgo'],
    zero_division=0
))

try:
    auc = roc_auc_score(y_c_test, y_churn_proba)
    print(f"  AUC-ROC: {auc:.4f}")
    if auc >= 0.85:
        print("  Interpretación: EXCELENTE discriminación")
    elif auc >= 0.70:
        print("  Interpretación: BUENO — útil para priorizar acciones")
    else:
        print("  Interpretación: MODERADO — revisar definición de churn")
except Exception as e:
    print(f"  AUC-ROC no calculable: {e}")
print("="*55)

# Matriz de confusión
cm = confusion_matrix(y_c_test, y_churn_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Activo', 'En riesgo']).plot(ax=ax)
plt.title('Matriz de Confusión — Modelo de Churn')
plt.tight_layout()
plt.show()

# ============================================================
# PASO 5: SALIDA ACCIONABLE — clientes en riesgo ordenados
# por probabilidad, para que el equipo comercial priorice
# ============================================================
resultado = test_c[['Channel', 'TipoProducto']].copy().reset_index(drop=True)
resultado['churn_pred']           = y_churn_pred
resultado['probabilidad_churn_%'] = (y_churn_proba * 100).round(1)

clientes_riesgo = (
    resultado[resultado['churn_pred'] == 1]
    .drop_duplicates(subset=['Channel', 'TipoProducto'])
    .sort_values('probabilidad_churn_%', ascending=False)
)

print(f"\nClientes-producto en riesgo detectados: {len(clientes_riesgo)}")
print("\nTop 15 — mayor probabilidad de churn:")
print(clientes_riesgo[['Channel', 'TipoProducto', 'probabilidad_churn_%']].head(15).to_string(index=False))

# Métricas solo sobre semanas con venta real (excluir ceros)
mask_activo = y_test > 0
mae_activo  = mean_absolute_error(y_test[mask_activo], y_pred[mask_activo])
r2_activo   = r2_score(y_test[mask_activo], y_pred[mask_activo])
print(f"MAE solo en semanas con venta : {mae_activo:.2f} unidades")
print(f"R²  solo en semanas con venta : {r2_activo:.4f}")

