# Proyecto 4 — Artículo de Análisis de Datos con CRISP-DM

Curso: Introducción a los Datos  
Docente: Hector Fabio Ocampo Arbeláez  
Modalidad: Grupal (mismos grupos del Proyecto 1, máximo 3 personas)  
Idioma del artículo: Español o Inglés

---

## Objetivo del Proyecto

Elaborar un artículo corto (paper) que documente el análisis de un problema real basado en datos, aplicando la metodología CRISP-DM. El artículo será presentado oralmente en la última clase del curso.

---

## Entregables

### 1. Artículo escrito

Un documento en formato PDF con la siguiente estructura (basada en un paper académico corto):

- **Título:** Descriptivo y conciso. Debe reflejar el problema y el enfoque.
- **Autores:** Nombres completos y correo institucional.
- **Resumen (Abstract):** 150–250 palabras. Problema, método, resultados principales y conclusión.
- **1. Introducción:** Contexto del problema, justificación (¿por qué importa?), y objetivo del análisis.
- **2. Comprensión del Negocio:** Pregunta(s) a responder, stakeholders, criterios de éxito del proyecto.
- **3. Comprensión de los Datos:** Origen del dataset, descripción de variables, estadísticas descriptivas, visualizaciones exploratorias.
- **4. Preparación de los Datos:** Limpieza (nulos, outliers), transformaciones (encoding, escalado), selección de variables.
- **5. Modelado:** Modelos aplicados (mínimo 2), justificación de la elección, hiperparámetros relevantes.
- **6. Evaluación:** Métricas utilizadas, comparación entre modelos, interpretación de resultados.
- **7. Conclusiones:** Hallazgos principales, limitaciones del estudio, recomendaciones y trabajo futuro.
- **Referencias:** Fuentes del dataset, librerías, y cualquier material consultado.

Extensión recomendada: 5–10 páginas (sin contar anexos de código).  
Formato: Letra 11pt, interlineado 1.15, márgenes 2.5cm. Se acepta cualquier template académico (IEEE, APA, o libre).

---

### 2. Código fuente

Notebook (.ipynb) o script (.py) con el código reproducible del análisis. Puede incluirse como anexo o enlace a repositorio (GitHub, Google Colab).

---

### 3. Presentación oral

Exposición de máximo 10 minutos en la última clase. Todos los integrantes deben participar. Se permite usar diapositivas de apoyo.

---

## Metodología: CRISP-DM

El proyecto debe seguir las fases de CRISP-DM (Cross-Industry Standard Process for Data Mining). No se requiere la fase de Despliegue (Deployment), pero sí se espera una reflexión sobre cómo se implementaría el modelo en producción.

1. **Business Understanding:** Definir el problema, los objetivos y los criterios de éxito.
2. **Data Understanding:** Explorar el dataset, identificar calidad y patrones iniciales.
3. **Data Preparation:** Limpiar, transformar y construir las variables necesarias para el modelado.
4. **Modeling:** Seleccionar y entrenar al menos 2 modelos. Justificar las decisiones.
5. **Evaluation:** Comparar modelos con métricas apropiadas al problema (clasificación, regresión o clustering).

---

## Selección del Dataset

Cada grupo debe elegir un dataset que cumpla con:

- Mínimo 1,000 registros.
- Al menos 8 variables (numéricas y categóricas).
- Un problema claramente definido (clasificación, regresión o clustering).
- No puede ser un dataset usado en clase (Iris, Titanic, Stroke, California Housing, MNIST).

### Fuentes sugeridas

- Kaggle Datasets — https://www.kaggle.com/datasets
- UCI Machine Learning Repository — https://archive.ics.uci.edu/
- Google Dataset Search — https://datasetsearch.research.google.com/
- Datos Abiertos Colombia — https://www.datos.gov.co/
- World Bank Open Data — https://data.worldbank.org/
- WHO Global Health Observatory — https://www.who.int/data/gho

---

## Evaluación

### Fechas importantes

- Elección del dataset (informar al docente)
- Entrega del artículo (PDF) y código: Última clase — antes de la presentación
- Presentación oral: Última clase del curso

### Criterios de evaluación

| Criterio | Peso | Detalle |
|---|---|---|
| Artículo — Estructura y redacción | 20% | Claridad, coherencia, ortografía, formato académico, abstract. |
| Artículo — Metodología CRISP-DM | 30% | Aplicación correcta de cada fase. Justificación de decisiones técnicas. |
| Artículo — Resultados y evaluación | 20% | Métricas apropiadas, comparación de modelos, visualizaciones, interpretación. |
| Presentación oral | 20% | Claridad, timing (≤10 min), dominio del tema, participación de todos. |
| Código reproducible | 10% | Notebook/script organizado, ejecutable, con comentarios mínimos. |

---

## Notas

- El artículo puede escribirse en inglés.
- Se penalizará el plagio. Citar correctamente cualquier fuente externa.
- Se valorará la originalidad en la elección del problema y la profundidad del análisis.
- No se requiere un modelo perfecto — se evalúa el proceso y la capacidad de análisis crítico.
