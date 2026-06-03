# Análisis Predictivo de Ventas e Inventarios en el Canal Distribuidor Samsung: Un enfoque basado en CRISP-DM, XGBoost y LSTM

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

## 1. Introducción

En el sector de tecnología de consumo, las decisiones de abastecimiento se toman muchas veces con base en la intuición o en registros del periodo anterior. Eso funciona cuando el mercado es estable, pero hoy los patrones de demanda cambian rápido. Un quiebre de stock en un producto de alta rotación puede significar perder no solo la venta, sino también al cliente.

Samsung Colombia lleva tiempo acumulando datos de ventas e inventarios por producto y por cliente distribuidor. El problema no es la falta de información, sino que esa información no se ha traducido en predicciones útiles para el negocio. Los datos están, pero no hablan.

Este análisis busca cambiar eso. Usando técnicas de machine learning y siguiendo la metodología CRISP-DM, tomamos los datos históricos disponibles y construimos modelos que permiten anticipar el comportamiento futuro de los indicadores clave: ventas por producto, tendencias por cliente y riesgo de desabastecimiento.

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

El proyecto se considera exitoso si los modelos logran predicciones con un error (RMSE o MAE) lo suficientemente bajo para ser útil en decisiones de abastecimiento. No buscamos modelos perfectos, sino que reduzcan la incertidumbre actual. Adicionalmente, el análisis debe identificar al menos dos clientes con tendencia de churn y dos productos con alta probabilidad de aumento en demanda.

---

## 3. Comprensión de los Datos

### 3.1 Origen del dataset

El dataset proviene del sistema interno de ventas de Samsung Colombia. Contiene registros históricos organizados en formato tabular ancho, donde cada columna representa una semana (en formato AAAA-SS, por ejemplo: 202301, 202302, etc.). Las filas representan combinaciones de cliente, producto y categoría de indicador. El dataset final, tras la transformación a formato longitudinal, contiene **4,206,567 registros**.

### 3.2 Variables principales

| Variable             | Tipo               | Descripción                                           |
| -------------------- | ------------------ | ------------------------------------------------------ |
| Channel              | Categórica        | Identificador del cliente distribuidor                 |
| Material Description | Categórica        | Descripción del producto con atributos concatenados   |
| Category             | Categórica        | Tipo de indicador: Sell-in, Cust. Sales o Channel Inv. |
| 202301...202352      | Numérica temporal | Valor del indicador por semana (formato wide)          |

### 3.3 Estadísticas descriptivas y patrones iniciales

Al explorar los datos se identificaron varios patrones relevantes:

**Distribución del valor:** La distribución es fuertemente sesgada a la derecha. Aproximadamente el 75% de los registros tiene valor cero, lo cual es coherente con un dataset de demanda intermitente: muchas combinaciones cliente-producto-semana no registran movimiento. Los valores extremos corresponden en su mayoría a clientes de alto volumen o a semanas de temporada alta.

**Comportamiento temporal por categoría:** El Channel Inv. domina en escala (promedio semanal agregado de aproximadamente 481,000 unidades), mientras que Cust. Sales y Sell-in se mueven en niveles más bajos (~64,000 y ~59,000 respectivamente). Las tres series presentan patrones propios y no se comportan de forma plana en el tiempo.

**Comportamiento semanal:** La semana 22 concentra el mayor volumen agregado (~2.4 millones) y la semana 52 el menor (~1.0 millón). Esto evidencia efectos estacionales que el modelo debe capturar.

**Concentración de clientes:** CUSTOMER20 (~17.5 M unidades) y CUSTOMER42 (~12.5 M) lideran con gran distancia sobre el resto. Pocos clientes explican una parte muy grande del movimiento total, lo que los convierte en clientes estratégicos cuya caída impacta significativamente el negocio.

**Portafolio de productos:** MOBILE domina ampliamente con ~50.9 M de unidades, más del doble que el segundo (LED TV, ~10.1 M). El portafolio está muy concentrado en móviles y televisores.

**Correlación con el tiempo:** Las correlaciones entre el Valor y las variables Año (~−0.015) o Semana (−0.19) son prácticamente nulas a nivel de registro individual, lo que confirma que el modelo necesita variables derivadas (lags, medias móviles) para capturar la dinámica temporal.

---

## 4. Preparación de los Datos

### 4.1 Transformación de estructura wide a long

El primer paso fue transformar el dataset de formato ancho (una columna por semana) a formato longitudinal, donde cada fila representa una combinación única de cliente, producto, categoría e indicador para una semana específica. Esto se hizo usando `pd.melt()` en Python. El resultado es una tabla con las columnas: Channel, Material Description, Category, Semana y Valor.

Este paso es fundamental porque los modelos de series de tiempo trabajan con datos longitudinales, no con tablas anchas.

### 4.2 Extracción de atributos del producto

La columna Material Description se procesó para extraer los atributos individuales del producto usando separadores de texto. Esto permitió crear columnas independientes para la línea de producto (TipoProducto), el modelo, el color y la capacidad de almacenamiento. Las columnas de texto resultantes se codificaron con Label Encoding para el modelado.

### 4.3 Tratamiento de nulos y valores atípicos

Los valores nulos se imputaron con cero cuando correspondían a semanas sin transacciones, validando que la ausencia de dato tenía significado comercial. Los outliers se detectaron con el método IQR y se analizaron caso por caso: algunos eran picos reales de demanda (Navidad, lanzamientos) y se conservaron; otros eran errores de carga y se corrigieron. Los valores negativos (generados por devoluciones o ajustes comerciales) se transformaron a cero antes del modelado, ya que el objetivo es predecir demanda real, no ajustes contables.

### 4.4 Construcción de features para el modelo

Para que XGBoost pueda aprender patrones temporales, se construyeron variables derivadas a partir de la semana: número de semana del año, mes, trimestre, y representaciones cíclicas mediante funciones seno y coseno (que permiten que el modelo entienda que la semana 1 y la 52 son temporalmente cercanas). Adicionalmente se crearon lags (valores de semanas anteriores) de 1, 2, 4, 8 y 12 semanas, medias móviles de 4, 8 y 12 semanas, y una medida de variabilidad (desviación estándar de 8 semanas). Para el modelo LSTM se aplicó además una transformación logarítmica (`log1p`) a los valores de venta, lo que reduce el impacto de valores extremos y mejora la estabilidad del entrenamiento.

| Feature                                      | Descripción                                    |
| -------------------------------------------- | ----------------------------------------------- |
| lag_1, lag_2, lag_4, lag_8, lag_12           | Valor del indicador en las semanas anteriores   |
| media_movil_4, media_movil_8, media_movil_12 | Promedio móvil de 4, 8 y 12 semanas            |
| std_8                                        | Desviación estándar de las últimas 8 semanas |
| semana_sin, semana_cos                       | Codificación cíclica de la semana del año    |
| Año, Mes, Semana                            | Variables temporales extraídas de la fecha     |
| Channel_enc                                  | Código numérico del cliente distribuidor      |

---

## 5. Modelado

### 5.1 Selección de modelos

Se aplicaron tres modelos con roles complementarios. La estrategia fue usar XGBoost como línea base robusta y luego LSTM para capturar la naturaleza secuencial de los datos de ventas.

**¿Por qué dos tipos de modelos?** XGBoost es un modelo tabular que aprende de features construidas a mano (lags, medias), mientras que LSTM aprende directamente de la secuencia temporal sin necesidad de que el analista construya todos los patrones de forma explícita. Comparar ambos enfoques permite identificar qué tipo de estructura captura mejor los patrones de este negocio.

### 5.2 Modelo 1 — XGBoost Regressor (predicción de volumen de ventas)

XGBoost (Extreme Gradient Boosting) es una implementación optimizada de Gradient Boosting: entrena árboles de decisión en secuencia, donde cada árbol corrige los errores del anterior. A diferencia de un árbol solo, que puede memorizar los datos de entrenamiento, el ensamblado de árboles con regularización incorporada generaliza mejor a datos nuevos. Maneja bien los datos tabulares con variables mixtas y es robusto ante valores nulos residuales.

El dataset se dividió respetando el orden temporal: entrenamiento con datos de 2023 y primeras 40 semanas de 2024; validación con las últimas 12 semanas de 2024; prueba con 2025. No se usó división aleatoria porque en series de tiempo el orden importa.

Los hiperparámetros principales del modelo final fueron:

- **n_estimators:** 1,000 árboles en el ensamblado
- **max_depth:** 8 niveles máximos por árbol
- **learning_rate:** 0.03 (tasa de aprendizaje conservadora)
- **subsample:** 0.8 (usa el 80% de las filas en cada árbol)
- **colsample_bytree:** 0.8 (usa el 80% de las features en cada árbol)
- **min_child_weight:** 5 (regularización mínima por nodo)

### 5.3 Modelo 2 — XGBoost Classifier (detección de churn)

Para responder la pregunta sobre clientes con tendencia a desaparecer, se construyó un problema de clasificación paralelo. Se definió churn como: un cliente-producto cuyas últimas 4 semanas disponibles en 2024 tienen valor de ventas igual a cero. Con esa etiqueta binaria (churn = 1 / activo = 0), se entrenó un segundo modelo XGBoost de clasificación.

El desbalance de clases (1.57 activos por cada cliente en riesgo) se manejó con el parámetro `scale_pos_weight`, lo que penaliza más los falsos negativos (no detectar un cliente que sí va a hacer churn). Los hiperparámetros utilizados: 300 árboles, learning_rate 0.05, max_depth 5, subsample 0.8, colsample_bytree 0.8.

### 5.4 Modelo 3 — LSTM con Embeddings (red neuronal secuencial)

Las redes LSTM (Long Short-Term Memory) son un tipo de red neuronal recurrente diseñada para aprender dependencias de largo plazo en secuencias. A diferencia de XGBoost, que recibe las semanas pasadas como columnas independientes, el LSTM recibe directamente la secuencia temporal y aprende a ponderar qué partes del pasado son más relevantes para la predicción actual, usando tres compuertas internas: la compuerta de olvido (decide qué información del pasado descartar), la compuerta de entrada (decide qué nueva información agregar) y la compuerta de salida (decide qué parte del estado interno exponer como predicción).

**Arquitectura del modelo (Multi-Input LSTM con Embeddings):**

El modelo tiene tres entradas simultáneas:

1. **Secuencia_Tiempo:** ventana de 8 semanas anteriores con 3 features cada una (Valor_Log, semana_sin, semana_cos).
2. **Canal:** identificador del cliente, procesado mediante un Embedding de dimensión 4.
3. **Producto:** identificador del tipo de producto, procesado mediante un Embedding de dimensión 16.

Los Embeddings permiten que la red aprenda una representación densa de cada cliente y cada producto, capturando similitudes entre ellos sin necesidad de codificación one-hot. El bloque LSTM tiene dos capas: la primera con 64 unidades y retorno de secuencias, seguida de Dropout(0.2); la segunda con 32 unidades. La salida del LSTM se concatena con los embeddings aplanados (dimensión total: 52) y pasa por una capa Dense(32, relu) con Dropout(0.2) antes de la predicción final.

| Capa             | Tipo                                      | Salida        | Parámetros             |
| ---------------- | ----------------------------------------- | ------------- | ----------------------- |
| Secuencia_Tiempo | InputLayer                                | (None, 8, 3)  | 0                       |
| lstm             | LSTM (64 unidades, return_sequences=True) | (None, 8, 64) | 17,408                  |
| dropout          | Dropout (0.2)                             | (None, 8, 64) | 0                       |
| lstm_1           | LSTM (32 unidades)                        | (None, 32)    | 12,416                  |
| Emb_Canal        | Embedding (dim=4)                         | (None, 1, 4)  | 392                     |
| Emb_Producto     | Embedding (dim=16)                        | (None, 1, 16) | 1,216                   |
| concatenate      | Concatenate                               | (None, 52)    | 0                       |
| dense            | Dense (32, relu)                          | (None, 32)    | 1,696                   |
| dropout_1        | Dropout (0.2)                             | (None, 32)    | 0                       |
| Prediccion_Log   | Dense (1, lineal)                         | (None, 1)     | 33                      |
| **Total**  |                                           |               | **33,161 params** |

El modelo se compiló con el optimizador Adam (learning_rate = 0.001), función de pérdida MSE y métrica MAE. La variable objetivo fue la venta en escala logarítmica (`log1p`), y las predicciones se invierten con `expm1` al momento de la evaluación. Se usó Early Stopping con paciencia de 10 épocas para evitar overfitting; el entrenamiento se detuvo en la época 24, restaurando los pesos de la época 14 (mejor resultado en validación). El dataset de entrenamiento contiene 135,354 secuencias.

---

## 6. Evaluación

### 6.1 Métricas utilizadas

Para los modelos de regresión (predicción de ventas) se usaron cinco métricas:

- **MAE (Error Absoluto Medio):** promedio del error en unidades reales. Es fácil de explicar al área comercial.
- **RMSE (Raíz del Error Cuadrático Medio):** penaliza errores grandes. Útil para detectar predicciones muy desviadas.
- **R² (Coeficiente de determinación):** qué porcentaje de la variabilidad de las ventas explica el modelo.
- **MAPE (Error Porcentual Absoluto Medio):** error relativo sobre registros con venta mayor a cero.
- **SMAPE (Error Porcentual Simétrico):** versión estabilizada del MAPE que evita divisiones por cero.

Para el modelo de clasificación de churn se usaron Precision, Recall, F1-Score y AUC-ROC. Se priorizó Recall sobre la clase "En riesgo" porque perder un cliente real de churn (falso negativo) tiene mayor costo comercial que alertar sobre uno que no iba a irse (falso positivo).

### 6.2 Resultados

**XGBoost Regressor — Predicción de Ventas (Cust. Sales)**

| Métrica | Resultado (todos los registros) | Resultado (solo semanas con venta > 0) |
| -------- | ------------------------------- | -------------------------------------- |
| MAE      | 32.3 unidades                   | 57.73 unidades                         |
| RMSE     | 260.93 unidades                 | 488.54 unidades                        |
| R²      | 0.7848 (78.5%)                  | 0.7919 (79.2%)                         |
| MAPE     | 130.67%                         | —                                     |
| SMAPE    | 122.21%                         | —                                     |

**XGBoost Classifier — Detección de Churn**

| Clase                     | Precision | Recall | F1-Score       | Support          |
| ------------------------- | --------- | ------ | -------------- | ---------------- |
| Activo                    | 0.94      | 0.88   | 0.91           | 20,097           |
| En riesgo                 | 0.77      | 0.88   | 0.82           | 8,976            |
| **Accuracy global** |           |        | **0.88** | **29,073** |

- **AUC-ROC: 0.9540** — Interpretación: EXCELENTE discriminación entre clientes activos y en riesgo.

**LSTM (V2 — Embeddings + Transformación Logarítmica)**

| Métrica             | Resultado                |
| -------------------- | ------------------------ |
| MAE                  | 0.05 (escala log)        |
| RMSE                 | 0.09 (escala log)        |
| R²                  | **0.8755 (87.6%)** |
| MAPE (sobre no-cero) | 32.06%                   |
| SMAPE                | 110.43%                  |

### 6.3 Interpretación de resultados

**Sobre el XGBoost de regresión:** El modelo alcanza un R² de 78.5%, lo que indica que explica cerca de cuatro quintas partes de la variabilidad en las ventas. El MAE de 32.3 unidades sobre el total de registros es razonable si se considera que la mayoría son ceros; sobre semanas con venta real, el error promedio sube a 57.7 unidades. El MAPE elevado (130.67%) se explica por la presencia de muchos registros con venta muy baja (1 o 2 unidades), donde cualquier error pequeño en valor absoluto se magnifica en términos porcentuales. Para la toma de decisiones comerciales, el MAE y el R² son las métricas más informativas.

El análisis de importancia de variables confirmó que los lags de corto y mediano plazo son las features más predictivas: lag_2 (15.7%), lag_4 (15.2%), Channel_enc (14.0%) y lag_1 (12.2%). Esto indica que el comportamiento reciente del cliente-producto es el mejor predictor de lo que va a pasar la semana siguiente, y que la identidad del canal tiene un peso casi tan alto como la historia de ventas. Las medias móviles también aportan significativamente, capturando la tendencia de mediano plazo.

**Sobre el clasificador de churn:** El AUC-ROC de 0.954 indica que el modelo tiene excelente capacidad para separar clientes activos de clientes en riesgo. El Recall de 0.88 sobre la clase "En riesgo" significa que el modelo detecta el 88% de los clientes que realmente van a hacer churn. Se identificaron **464 combinaciones cliente-producto en riesgo**, con los siguientes casos de mayor probabilidad:

| Cliente    | Tipo de Producto   | Probabilidad de Churn |
| ---------- | ------------------ | --------------------- |
| CUSTOMER12 | SIGNAGE_STANDALONE | 99.8%                 |
| CUSTOMER12 | BUSINESS TV        | 99.8%                 |
| CUSTOMER12 | LFD                | 99.6%                 |
| CUSTOMER97 | MOBILE             | 99.4%                 |
| CUSTOMER3  | CAC                | 98.9%                 |
| CUSTOMER25 | TABLET             | 98.1%                 |
| CUSTOMER30 | LED TV             | 97.6%                 |

Los clientes identificados en riesgo comparten un patrón: redujeron su Cust. Sales de forma sostenida durante las últimas semanas antes de llegar a cero, lo que significa que el modelo puede detectar la señal antes de que el área comercial la vea en los pedidos.

**Sobre el LSTM:** El LSTM alcanzó el mejor R² de los tres modelos (87.6%), superando al XGBoost de regresión en casi 9 puntos porcentuales. Esto indica que la arquitectura secuencial captura mejor los patrones de la demanda que el enfoque basado en features tabulares. El MAPE de 32.06% sobre registros con venta positiva es considerablemente mejor que el del XGBoost (130.67%), lo que confirma que el LSTM hace predicciones más ajustadas en los periodos de venta activa. El SMAPE elevado (110.43%) se debe al mismo efecto de los ceros: cuando la predicción y el valor real son ambos cercanos a cero, el denominador se hace muy pequeño y la métrica se distorsiona.

---

## 7. Conclusiones

### 7.1 Respuestas a las preguntas estratégicas del negocio

**Pregunta 1: ¿Qué producto tendrá mayor rotación en los próximos meses?**

El producto con mayor rotación proyectada es **MOBILE**, que concentra aproximadamente el 63% del volumen total de Cust. Sales (~50.9 M unidades históricas) y mantiene crecimiento sostenido en los datos de 2024–2025. En particular, los modelos de gama media dentro de la categoría MOBILE muestran crecimiento en Cust. Sales independientemente de la temporada, lo que los convierte en los productos con mayor probabilidad de mantener alta rotación en los próximos meses.

**Pregunta 2: ¿Cuáles clientes tienen tendencia a desaparecer?**

El clasificador de churn identificó 464 combinaciones cliente-producto en riesgo. Los clientes con mayor probabilidad de abandono son **CUSTOMER12** (con riesgo superior al 99.6% en líneas de negocio B2B como SIGNAGE y BUSINESS TV), **CUSTOMER97** (riesgo del 99.4% en MOBILE), **CUSTOMER3** (riesgo superior al 98% en múltiples categorías) y **CUSTOMER25** (98.1% en TABLET). El modelo detecta esta señal con semanas de anticipación respecto al momento en que la caída sería visible en los pedidos de Sell-in.

**Pregunta 3: ¿Qué productos incrementarán sus ventas y qué clientes las reducirán?**

Los productos con tendencia de incremento son los de gama media en la categoría MOBILE, que muestran crecimiento sostenido en Cust. Sales. LED TV también muestra recuperación en algunos clientes clave. Por el contrario, las categorías BLU-RAY DISC PLAYER y RSD presentan señales de contracción acelerada, con múltiples clientes clasificados en riesgo de churn para esas líneas. En cuanto a clientes, los que el modelo proyecta con reducción más significativa son CUSTOMER12, CUSTOMER97, CUSTOMER3 y CUSTOMER30, quienes tienen alta probabilidad de churn en varias categorías simultáneamente.

**Pregunta 4: ¿A qué cliente se le debería despachar más producto y cuál específicamente?**

Con base en el volumen histórico y la proyección del modelo, los clientes con mayor demanda sostenida son **CUSTOMER20** y **CUSTOMER42**, que lideran el ranking de Cust. Sales total (~17.5 M y ~12.5 M unidades respectivamente). El modelo de regresión proyecta que estos clientes mantendrán su nivel de demanda, por lo que son prioritarios para garantizar inventario suficiente. El producto específico a despachar con mayor urgencia es **MOBILE** (especialmente para CUSTOMER20 y CUSTOMER42), seguido de **LED TV** para los clientes de mediano volumen que no muestran señal de churn.

### 7.2 Hallazgos principales

El análisis confirmó que los datos históricos de Samsung Colombia contienen señales suficientes para anticipar el comportamiento futuro de ventas e inventarios. El modelo LSTM con embeddings logró el mejor desempeño general (R² = 87.6%), demostrando que la estructura secuencial de la demanda semanal aporta información que los modelos tabulares no capturan completamente. El clasificador de churn (AUC-ROC = 0.954) identificó con alta precisión los clientes en riesgo de abandono, con un Recall del 88% sobre la clase de interés.

### 7.3 Limitaciones del estudio

El modelo actual no incorpora variables externas que pueden impactar significativamente las ventas: lanzamientos de nuevos modelos, campañas de marketing, variaciones del precio del dólar o acciones de la competencia. Incluir estas variables en una siguiente versión del modelo probablemente mejoraría la precisión.

El MAPE y SMAPE elevados de ambos modelos de regresión se deben a la alta proporción de ceros en el dataset (demanda intermitente). Para productos de alta frecuencia, las métricas son considerablemente mejores. Para productos de baja frecuencia de compra, se recomienda complementar con un modelo de clasificación que primero estime si habrá venta o no, y luego un modelo de regresión que estime cuánto.

Adicionalmente, la definición de churn usada (últimas 4 semanas con valor cero en 2024) es una aproximación. En un proyecto de mayor alcance, se debería validar esta definición con el área comercial para asegurarse de que corresponde a lo que ellos consideran un cliente en riesgo real.

### 7.4 Recomendaciones y trabajo futuro

Se recomienda implementar el modelo en un ciclo de actualización semanal, donde cada lunes se re-entrene con los datos de la semana anterior y se genere un reporte con las predicciones y alertas de churn para la semana en curso. Esto requiere un pipeline automatizado, pero la lógica ya está desarrollada en el notebook entregado.

Como trabajo futuro, se sugiere explorar modelos de demanda intermitente como Croston o sus variantes, más adecuados para combinaciones cliente-producto con alta proporción de ceros. También se recomienda construir un dashboard interactivo que le permita al equipo comercial consultar las predicciones por cliente y producto sin necesidad de acceder al notebook, y explorar el uso de variables exógenas (precio, competencia, campañas) para mejorar la precisión de las predicciones de largo plazo.

---

## Referencias

Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). CRISP-DM 1.0: Step-by-step data mining guide. SPSS Inc.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.

McKinney, W. (2017). Python for Data Analysis (2nd ed.). O'Reilly Media.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

Abadi, M., et al. (2016). TensorFlow: A system for large-scale machine learning. 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16), 265–283.

Dataset: Datos internos de ventas e inventarios Samsung Colombia, proporcionados por la dirección del área para fines académicos.
