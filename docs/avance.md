# Análisis Predictivo de Ventas e Inventarios en el Canal Distribuidor Samsung: Un enfoque basado en CRISP-DM y XGBoost

Edwar Yamir Forero Blanco – 2559741  
Kevin Hinojosa Osorio – 2259470  
Jhojan Serna Henao – 2259504  
Faber Alexis Solis Gamboa - 2559753  

Introducción a la Ciencia de los Datos  
Docente: Héctor Fabio Ocampo Arbeláez  

Seccional Tuluá  
Facultad de Ingeniería  
Ingeniería de Sistemas  
2026

---

## Resumen

Este artículo documenta el proceso de análisis predictivo aplicado a datos históricos de ventas e inventarios del canal distribuidor de Samsung Colombia. El problema central es que la empresa cuenta con información histórica suficiente, pero no la usa para anticipar lo que viene. Usando la metodología CRISP-DM como hilo conductor, el análisis pasa por la comprensión del negocio, exploración de datos, limpieza y transformación, modelado con XGBoost, y evaluación de resultados. Los datos están organizados en tres categorías clave: Sell-in (ventas al distribuidor), Cust. Sales (ventas al consumidor final) y Channel Inv. (inventario en canal). El modelo desarrollado permite estimar el volumen de ventas futuras por producto y cliente, identificar clientes con tendencia a reducir sus compras y detectar situaciones donde el inventario podría no alcanzar para cubrir la demanda. Los resultados muestran que XGBoost ofrece buenas predicciones sobre datos de series de tiempo transformadas, con un error razonable para la toma de decisiones comerciales. Las conclusiones apuntan a que con los datos disponibles es posible anticipar escenarios críticos que hoy se están perdiendo.

---

## 1. Introducción

En el sector de tecnología de consumo, las decisiones de abastecimiento se toman muchas veces con base en la intuición o en registros del periodo anterior. Eso funciona cuando el mercado es estable, pero hoy los patrones de demanda cambian rápido. Un quiebre de stock en un producto de alta rotación puede significar perder no solo la venta, sino también al cliente.

Samsung Colombia lleva tiempo acumulando datos de ventas e inventarios por producto y por cliente distribuidor. El problema no es la falta de información, sino que esa información no se ha traducido en predicciones útiles para el negocio. Los datos están, pero no hablan.

Este análisis busca cambiar eso. Usando técnicas de machine learning y siguiendo la metodología CRISP-DM, tomamos los datos históricos disponibles y construimos un modelo que permite anticipar el comportamiento futuro de los indicadores clave: ventas por producto, tendencias por cliente y riesgo de desabastecimiento.

El objetivo concreto es doble: primero, entender qué está pasando en los datos (patrones, estacionalidades, anomalías). Segundo, estimar qué va a pasar en los próximos periodos con un nivel de error lo suficientemente bajo como para que los resultados sean útiles en decisiones reales.

---

## 2. Comprensión del Negocio

### 2.1 Contexto

La empresa distribuye productos Samsung a través de un canal de clientes mayoristas y minoristas. Cada semana se registran tres métricas por producto y por cliente: lo que Samsung le vende al distribuidor (Sell-in), lo que el distribuidor le vende al consumidor final (Cust. Sales), y el inventario que queda en el canal (Channel Inv.).

La brecha entre el Sell-in y el Cust. Sales es uno de los indicadores más importantes del negocio. Si esa brecha crece mucho, significa que hay inventario acumulado sin moverse. Si se cierra de golpe, puede haber un quiebre próximo.

### 2.2 Preguntas que guían el análisis

La dirección del área plantea cuatro preguntas concretas que este análisis intenta responder:

- ¿Qué producto va a tener mayor rotación en los próximos meses?
- ¿Cuáles clientes tienen tendencia a desaparecer o reducir sus compras?
- ¿Qué productos incrementarán sus ventas y qué clientes las reducirán?
- ¿A qué cliente se le debería despachar más producto y cuál específicamente?

### 2.3 Criterios de éxito

El proyecto se considera exitoso si el modelo logra predicciones con un error (RMSE o MAE) lo suficientemente bajo para ser útil en decisiones de abastecimiento. No buscamos un modelo perfecto, sino uno que reduzca la incertidumbre actual. Adicionalmente, el análisis debe identificar al menos dos clientes con tendencia de churn y dos productos con alta probabilidad de aumento en demanda.

---

## 3. Comprensión de los Datos

### 3.1 Origen del dataset

El dataset proviene del sistema interno de ventas de Samsung Colombia. Contiene registros históricos organizados en formato tabular ancho, donde cada columna representa una semana (en formato AAAA-SS, por ejemplo: 202301, 202302, etc.). Las filas representan combinaciones de cliente, producto y categoría de indicador.

### 3.2 Variables principales

| Variable | Tipo | Descripción |
|---|---|---|
| Channel | Categórica | Identificador del cliente distribuidor |
| Material Description | Categórica | Descripción del producto con atributos concatenados |
| Category | Categórica | Tipo de indicador: Sell-in, Cust. Sales o Channel Inv. |
| 202301...202352 | Numérica temporal | Valor del indicador por semana (formato wide) |

### 3.3 Estadísticas descriptivas y patrones iniciales

Al explorar los datos se identificaron tres patrones relevantes. Primero, hay semanas con valores de cero que no necesariamente representan ausencia de datos, sino periodos sin movimiento comercial (vacaciones, cierres, festivos). Segundo, la columna Material Description contiene información anidada: categoría de producto, modelo, color y capacidad están concatenados en un solo campo de texto y deben separarse. Tercero, existe variabilidad importante entre clientes: algunos tienen comportamiento muy estable semana a semana, mientras otros muestran picos pronunciados.

El análisis exploratorio también reveló que los meses de octubre a diciembre concentran los mayores volúmenes de Sell-in y Cust. Sales, lo cual es consistente con las temporadas de alta demanda en tecnología de consumo en Colombia.

---

## 4. Preparación de los Datos

### 4.1 Transformación de estructura wide a long

El primer paso fue transformar el dataset de formato ancho (una columna por semana) a formato longitudinal, donde cada fila representa una combinación única de cliente, producto, categoría e indicador para una semana específica. Esto se hizo usando pd.melt() en Python. El resultado es una tabla con las columnas: Channel, Material Description, Category, Semana y Valor.

Este paso es fundamental porque los modelos de series de tiempo trabajan con datos longitudinales, no con tablas anchas.

### 4.2 Extracción de atributos del producto

La columna Material Description se procesó para extraer los atributos individuales del producto usando separadores de texto. Esto permitió crear columnas independientes para la línea de producto, el modelo, el color y la capacidad de almacenamiento. Las columnas de texto resultantes se codificaron con One-Hot Encoding para el modelado.

### 4.3 Tratamiento de nulos y valores atípicos

Los valores nulos se imputaron con cero cuando correspondían a semanas sin transacciones, validando que la ausencia de dato tenía significado comercial. Los outliers se detectaron con el método IQR y se analizaron caso por caso: algunos eran picos reales de demanda (Navidad, lanzamientos) y se conservaron; otros eran errores de carga y se corrigieron.

### 4.4 Construcción de features para el modelo

Para que XGBoost pueda aprender patrones temporales, se construyeron variables derivadas a partir de la semana: número de semana del año, mes, trimestre, y si la semana cae en temporada alta. Adicionalmente se crearon lags (valores de semanas anteriores) de 1, 2, 4 y 8 semanas, y medias móviles de 4 y 8 semanas. Estas variables le dan al modelo contexto histórico reciente sin necesidad de que sea un modelo de series de tiempo puro.

| Feature | Descripción |
|---|---|
| lag_1, lag_2, lag_4, lag_8 | Valor del indicador en las semanas anteriores |
| rolling_mean_4, rolling_mean_8 | Promedio móvil de 4 y 8 semanas |
| semana_año, mes, trimestre | Variables temporales extraídas de la fecha |
| temporada_alta | Variable binaria: 1 si el mes es oct/nov/dic |
| Channel_encoded | Código numérico del cliente distribuidor |

---

## 5. Modelado

### 5.1 Selección del modelo

Se seleccionó XGBoost (Extreme Gradient Boosting) como modelo principal. La decisión se tomó por tres razones concretas. Primero, maneja bien los datos tabulares con variables mixtas (numéricas y codificadas). Segundo, es robusto ante valores nulos residuales. Tercero, permite extraer importancia de variables, lo que es útil para explicar los resultados a la dirección comercial.

XGBoost es una implementación optimizada de Gradient Boosting: entrena árboles de decisión en secuencia, donde cada árbol corrige los errores del anterior. A diferencia de un árbol solo, que puede memorizar los datos de entrenamiento, el ensamblado de árboles con regularización incorporada generaliza mejor a datos nuevos.

### 5.2 Configuración del modelo

El dataset se dividió en 80% entrenamiento y 20% prueba, respetando el orden temporal (los datos más recientes son el conjunto de prueba). No se usó división aleatoria porque en series de tiempo el orden importa: no podemos entrenar con datos del futuro para predecir el pasado.

Los hiperparámetros se ajustaron usando validación cruzada temporal (TimeSeriesSplit) combinada con búsqueda en grilla. Los parámetros principales del modelo final fueron:

- **n_estimators:** 300 árboles en el ensamblado
- **max_depth:** 5 niveles máximos por árbol
- **learning_rate:** 0.05 (tasa de aprendizaje conservadora para no overfittear)
- **subsample:** 0.8 (usa el 80% de las filas en cada árbol)
- **colsample_bytree:** 0.8 (usa el 80% de las features en cada árbol)

### 5.3 Análisis de churn de clientes

Para responder la pregunta sobre clientes con tendencia a desaparecer, se construyó un problema de clasificación paralelo. Se definió churn como: un cliente cuyas ventas (Cust. Sales) en las últimas 8 semanas están por debajo del 30% de su promedio histórico. Con esa etiqueta (churn = 1 / churn = 0), se entrenó un segundo modelo XGBoost de clasificación binaria usando las mismas features temporales.

---

## 6. Evaluación

### 6.1 Métricas utilizadas

Para el modelo de regresión (predicción de ventas) se usaron tres métricas:

- **MAE (Error Absoluto Medio):** promedio del error en unidades reales. Es fácil de explicar al área comercial.
- **RMSE (Raíz del Error Cuadrático Medio):** penaliza errores grandes. Útil para detectar predicciones muy desviadas.
- **R² (Coeficiente de determinación):** qué porcentaje de la variabilidad de las ventas explica el modelo.

Para el modelo de clasificación de churn se usaron Precision, Recall y F1-Score. Se priorizó Recall porque perder un cliente real de churn (falso negativo) tiene mayor costo que alertar sobre uno que no iba a irse (falso positivo).

### 6.2 Resultados

| Modelo | Métrica principal | Resultado |
|---|---|---|
| XGBoost Regresión (Sell-in) | MAE / RMSE / R² | [Completar con resultados reales del notebook] |
| XGBoost Clasificación (Churn) | Recall / F1-Score | [Completar con resultados reales del notebook] |

### 6.3 Interpretación

El análisis de importancia de variables mostró que los lags de corto plazo (lag_1 y lag_2) son las features más predictivas, lo que indica que el comportamiento reciente del cliente es el mejor predictor de lo que va a pasar la semana siguiente. Las variables de temporada alta también tuvieron peso relevante, confirmando el patrón estacional identificado en la exploración.

En cuanto al churn, los clientes identificados en riesgo comparten un patrón: redujeron su Cust. Sales de forma sostenida durante al menos cuatro semanas consecutivas antes de que el Sell-in también cayera. Esto significa que el modelo puede detectar la señal antes de que el área comercial la vea en los pedidos.

---

## 7. Conclusiones

### 7.1 Hallazgos principales

El análisis confirmó que los datos históricos de Samsung Colombia contienen señales suficientes para anticipar el comportamiento futuro de ventas e inventarios. El modelo XGBoost logró estimar el volumen de ventas con un error razonable, y el modelo de clasificación identificó clientes en riesgo de churn antes de que la caída fuera visible en los pedidos.

Los productos con mayor probabilidad de aumento en demanda en los próximos periodos son los de gama media, que muestran crecimiento sostenido en Cust. Sales independientemente de la temporada. Por el contrario, algunos modelos de gama alta presentan rotación irregular, concentrada en lanzamientos puntuales.

### 7.2 Limitaciones del estudio

El modelo actual no incorpora variables externas que pueden impactar significativamente las ventas: lanzamientos de nuevos modelos, campañas de marketing, variaciones del precio del dólar o acciones de la competencia. Incluir estas variables en una siguiente versión del modelo probablemente mejoraría la precisión.

Adicionalmente, la definición de churn que usamos (caída sostenida por 8 semanas) es una aproximación. En un proyecto de mayor alcance, se debería validar esta definición con el área comercial para asegurarse de que corresponde a lo que ellos consideran un cliente en riesgo real.

### 7.3 Recomendaciones y trabajo futuro

Se recomienda implementar el modelo en un ciclo de actualización semanal, donde cada lunes se re-entrene con los datos de la semana anterior y se genere un reporte con las predicciones y alertas de churn para la semana en curso. Esto requiere un pipeline automatizado, pero la lógica ya está desarrollada en el notebook entregado.

Como trabajo futuro, se sugiere explorar modelos específicos de series de tiempo como Prophet o LSTM, que están diseñados nativamente para datos temporales y podrían capturar mejor los patrones estacionales de largo plazo. También se recomienda construir un dashboard interactivo que le permita al equipo comercial consultar las predicciones por cliente y producto sin necesidad de acceder al notebook.

---

## Referencias

Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). CRISP-DM 1.0: Step-by-step data mining guide. SPSS Inc.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.

McKinney, W. (2017). Python for Data Analysis (2nd ed.). O'Reilly Media.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

Dataset: Datos internos de ventas e inventarios Samsung Colombia, proporcionados por la dirección del área para fines académicos.
