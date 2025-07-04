# 🦠 COVID-19 Predictive Pipeline with Concept Drift Adaptation

Pipeline MLOps desarrollada con **ZenML** para entrenar y actualizar de forma continua un modelo predictivo sobre datos de pacientes con COVID. El objetivo es detectar y adaptarse automáticamente al **concept drift** en los datos de estratificación ofrecidos por SMS, asegurando un desempeño óptimo en el tiempo.

---

## 📌 Introducción

El **concept drift** se refiere a cambios en la relación entre las variables de entrada (X) y la variable objetivo (y) a lo largo del tiempo, lo que provoca que un modelo previamente entrenado pierda precisión.

Nuestra pipeline detecta el drift de forma **implícita**, evaluando mensualmente el rendimiento sobre datos nuevos (chunks mensuales) y comparándolo con el desempeño de modelos anteriores. Si el rendimiento cae (por ejemplo, baja la balanced accuracy), la pipeline evita actualizar el modelo, asegurando que solo se reentrene cuando el nuevo modelo realmente mejora o mantiene el desempeño.

---

## 💡 Idea Principal

1. **Monitoreo de desempeño:**  
   Evalúa el rendimiento en los chunks más recientes respecto a modelos anteriores usando métricas como *balanced accuracy*.

2. **Reentrenamiento adaptativo:**  
   Si los datos cambian, la pipeline entrena ensembles actualizados con los datos más recientes, ajustando el modelo al nuevo patrón.

3. **Automatización con CRON:**  
   La ejecución automática mediante `crontab` garantiza un sistema continuo de aprendizaje y actualización.

---

## 🔄 Flujo de Trabajo

### 1️⃣ Ingesta de Datos

- Carga los datos y verifica la fecha de última modificación del archivo.
- Si no hubo cambios desde la última ejecución (mediante un archivo de control), la pipeline termina para optimizar recursos.
- Si se detectan cambios, devuelve el dataframe actualizado.

### 2️⃣ Preprocesamiento de Datos

- Elimina nulos en columnas clave.
- Genera variables binarias (e.g., `OBESITY`, `ASTHMA`) a partir de la columna `ETIQUETA`.
- Procesa la fecha de diagnóstico para extraer el mes (`mes`) y segmentar los datos en **chunks mensuales**.
- Convierte variables categóricas como `INGRESO` y `SEXO` a formato numérico.

### 3️⃣ Entrenamiento

- Divide los datos en chunks mensuales, funcionando como mini-datasets independientes.
- Genera múltiples bootstraps balanceados con la estrategia **IPIP** (Iterative Proportional Importance Pruning).
- Cada bootstrap entrena un ensemble de Random Forests, manteniendo solo los modelos que mejoran el desempeño del ensemble en un conjunto de validación.
- Genera predicciones sobre el chunk siguiente, simulando la predicción en datos futuros.
- Guarda resultados y comparativas de métricas con modelos previos.

### 4️⃣ Validación

- Calcula métricas globales y por chunk: *accuracy*, *precision*, *recall*, *f1* y *balanced accuracy*.
- Genera gráficos de evolución temporal de la balanced accuracy y matriz de confusión acumulada.
- Guarda métricas y gráficos como evidencias de monitoreo.

### 5️⃣ Despliegue de Modelos

- Compara el rendimiento del nuevo modelo con el anterior.
- Solo guarda el nuevo modelo si:
  - El rendimiento supera el umbral mínimo (balanced accuracy ≥ 0.9), **y**
  - El modelo mejora o mantiene el desempeño frente al modelo previo.
- Guarda ejemplos de predicciones para auditoría y trazabilidad.

---

## 🔍 Concept Drift

El enfoque se basa en segmentar los datos en chunks mensuales para detectar y responder rápidamente a cambios en su comportamiento. Para cada nuevo chunk, se compara el rendimiento del modelo actual con modelos anteriores, funcionando como un test supervisado de drift.

La pipeline evita sobreescribir el modelo si el nuevo no muestra una mejora significativa, previniendo degradaciones por drift temporal o ruido. La metodología **IPIP** mejora además el manejo de conjuntos de datos desbalanceados.

---

## 🧩 IPIP (Iterative Proportional Importance Pruning)

La técnica **IPIP** combina bagging, ensembles y balanceo iterativo para mejorar la predicción en problemas con clases desbalanceadas y series temporales. Permite:

- Equilibrar clases en cada bootstrap.
- Seleccionar modelos que realmente aportan mejoras al ensemble.
- Adaptar el modelo dinámicamente a cambios en la distribución de los datos (concept drift).

---

## 🚀 Automatización

Integra tareas programadas con `CRONTAB` para ejecutar la pipeline periódicamente, garantizando un sistema de aprendizaje continuo y sin intervención manual.

---

## 🗂️ Tecnologías

- ZenML
- Python 3.10  
- Scikit-learn  
- Pandas  
- Matplotlib / Seaborn  
- CRON para automatización

---
