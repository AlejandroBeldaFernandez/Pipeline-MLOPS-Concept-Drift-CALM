# Introducción
Pipeline MLOPS desarrollada con ZenML la cuál implementa un sistema de entrenamiento y actualización de un modelo predictivo sobre datos de pacientes con COVID, con el objetivo de  detectar y adaptarse al concept drift en los datos de estratificación ofrecidos por SMS. Entendemos el Concept Drift como los cambios en la relación entre las variables de entrada (X) y la variable objetivo (y) a lo largo del tiempo, lo que provoca que un modelo previamente entrenado pierda precisión. Aquí, la  pipeline detecta concept drift implícitamente evaluando mensualmente el rendimiento sobre datos nuevos (chunks mensuales) y comparándolo con modelos anteriores. Si el rendimiento cae (balanced accuracy empeora), evita actualizar el modelo, asegurando que solo se reentrene cuando el nuevo modelo realmente mejora o mantiene el desempeño.

# Idea Princpial 
1. Monitoreo de drift a nivel de desempeño: La pipeline evalúa si la performance en los chunks más recientes disminuye respecto a modelos anteriores, usando métricas como balanced accuracy.

2. Reentrenamiento adaptativo: Si los datos cambian, el pipeline entrena ensembles actualizados con los datos más recientes, generando modelos que se ajustan al posible nuevo patrón en los datos.

3. Automatización con CRON: Ejecutar la  pipeline automáticamente mediante CRONTAB lo que  garantiza un sistema continuo de aprendizaje y actualización automatica.

# Flujo de Trabajo 

1. data_loader

Carga los datos y comprueba la fecha de última modificación del archivo; si no hubo cambios desde la última ejecución (usando un archivo de control), termina el pipeline sin continuar, optimizando recursos y finalmente s devuelve el dataframe si detecta cambios (potencial concept drift).

2. data_preprocessing

Limpia los datos eliminando nulos en columnas clave. Genera variables binarias (e.g., OBESITY, ASTHMA)  a partir del texto de ETIQUETA. Procesar la fecha de diagnóstico para extraer el mes (columna mes), lo que permite segmentar los datos en "chunks" mensuales. Convertir variables categóricas como INGRESO y SEXO en formato numérico.

3. trainer

Divide los datos por chunks (meses) y usa los chunks como mini-datasets independientes.Para cada chunk, genera múltiples bootstraps balanceando las clases (estrategia IPIP: Iterative Proportional Importance Pruning). Cada bootstrap entrena un ensemble de Random Forests con selección iterativa de modelos: solo se mantienen modelos que mejoran el desempeño del ensemble sobre un conjunto de validación. Genera predicciones sobre el chunk siguiente, simulando predicción en datos futuros.Guarda resultados y comparativas de métricas con modelos anteriores.

4. evaluacion

Calcula métricas globales y por chunk: accuracy, precision, recall, f1 y balanced accuracy.

Genera gráficos de evolución temporal del balanced accuracy y matriz de confusión total.

Guarda métricas y gráficos como evidencias de monitoreo.

5. save_model

Compara el rendimiento del modelo actual con el modelo previo.Solo guarda los nuevos modelos si el rendimiento supera un umbral (balanced accuracy ≥0.9) y es mejor que el modelo anterior. También guarda ejemplos de predicciones para auditoría.

# Concept Drift
La segmentación temporal en chunks permite detectar y responder rápidamente a cambios en el comportamiento de los datos. La evaluación del rendimiento en cada chunk nuevo compara modelos pasados y actuales, lo que funciona como un test de drift supervisado. La pipeline evita sobreescribir modelos cuando el nuevo modelo no mejora (previniendo degradaciones debido a drift temporal o ruido).

# IPIP 
Utilizacion de la tecnica IPIP, la cual es una  estrategia que combina bagging, ensembles y balanceo iterativo para mejorar la predicción en problemas con clases desbalanceadas y series temporales, evaluando chunk a chunk para detectar y adaptarse a posibles cambios en la distribución de los datos (concept drift).

