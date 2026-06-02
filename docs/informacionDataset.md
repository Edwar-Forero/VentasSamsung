## 1. OBJETIVO PRINCIPAL
El objetivo principal no es solo aplicar algoritmos, sino comprender el contexto comercial antes de iniciar cualquier proceso de automatización. El análisis predictivo permite utilizar datos históricos para estimar comportamientos futuros del negocio, lo cual tiene un impacto directo en las decisiones comerciales y operativas para anticipar escenarios, evitar quiebres de stock y prevenir la pérdida de clientes.

---

## 2. FORMATO DE ENTREGABLES
*(Aclaración del Reto)*

### 💻 Notebook Analítico:
En lugar de un dashboard interactivo tradicional o un software independiente, el análisis debe presentarse mediante un **Notebook de Python** (o entorno similar) bien estructurado. Este documento debe contener el código fuente, la lógica de comprensión de los datos y gráficos integrados que guíen el hilo conductor del análisis de principio a fin.

### 📄 Documento de Presentación:
Un reporte analítico detallado dirigido a la dirección de ventas (Canva, PowerPoint, PDF, etc.). Las presentaciones deben mantener un diseño gráfico sobrio y minimalista para generar un impacto real.

---

## 3. COMPRENSIÓN DE LAS ESTRUCTURAS DE DATOS

* **👥 Variables Categoricas:**
    Incluyen el identificador del cliente (`Channel`) y la descripción del producto (`Material Description`), la cual contiene múltiples atributos concatenados que deben ser procesados.
* **📅 Variables Numéricas Temporales:**
    Los datos están distribuidos en columnas semanales (ej. `202301`, `202302`, etc.).
* **📊 Indicadores Clave (Category):**
    Es vital que los equipos comprendan la dinámica de la cadena de suministro representada en la columna `Category`, la cual divide las métricas en:
    * Ventas al canal distribuidor (**Sell-in**)
    * Ventas al consumidor final (**Cust. Sales**)
    * Inventario disponible (**Channel Inv.**)

---

## 4. HILO CONDUCTOR DEL ANÁLISIS (FASES DEL NOTEBOOK)
El desarrollo técnico esperado dentro del código debe evidenciar las siguientes etapas:

### 🅰️ PROCESAMIENTO E INGENIERÍA DE CARACTERÍSTICAS
* Transformar la estructura tabular ancha (semanas en columnas) a un formato longitudinal apropiado para el modelado de series de tiempo.
* Extraer y limpiar la información anidada dentro de las descripciones de los materiales.

### 🅱️ ANÁLISIS ESTADÍSTICO Y PROBABILIDAD
* Aplicar conceptos estadísticos básicos para comprender el comportamiento de los datos y validar los hallazgos.
* Calcular el promedio de ventas mensuales y anuales, identificar los mejores meses de ventas y analizar la rotación de inventarios a nivel general y por punto específico.

### Ⓒ MODELADO PREDICTIVO (MACHINE LEARNING)
* Desarrollar un modelo de aprendizaje supervisado que logre estimar el comportamiento futuro de los indicadores clave.
* El modelo debe dar respuesta cuantitativa a las siguientes preguntas estratégicas:
    1. ¿Qué producto tendrá mayor rotación en los próximos meses?
    2. ¿Cuáles clientes tienen tendencia a desaparecer (abandono/churn)?
    3. ¿Qué productos incrementarán sus ventas y qué clientes las reducirán?
    4. ¿A qué cliente se le debería despachar más producto y cuál específicamente?