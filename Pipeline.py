from zenml import pipeline, step
from typing import Annotated, Tuple, List, Optional
from sklearn.base import ClassifierMixin
import pandas as pd 
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import logging  # Gestión de alertas, warnings de la pipeline (corregido)
from pathlib import Path

def setup_pipeline_logging():
    """
    Configura el sistema de logging específico para la pipeline
    """
    # Crear directorio de logs
    log_dir = Path('/opt/covid/logs_Estratificacion')
    log_dir.mkdir(exist_ok=True)
    
    # Configurar formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Logger principal de la pipeline
    logger = logging.getLogger('covid_pipeline')
    logger.setLevel(logging.DEBUG)
    
    # Limpiar handlers existentes
    logger.handlers.clear()
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Handler para archivo general
    file_handler = logging.FileHandler(log_dir / 'pipeline.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Handler específico para errores
    error_handler = logging.FileHandler(log_dir / 'pipeline_errors.log', encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    
    # Handler específico para warnings
    warning_handler = logging.FileHandler(log_dir / 'pipeline_warnings.log', encoding='utf-8')
    warning_handler.setLevel(logging.WARNING)
    warning_handler.setFormatter(formatter)
    
    # Añadir handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(warning_handler)
    
    return logger

pipeline_logger = setup_pipeline_logging()

@step(enable_cache=False)
def data_loader() -> pd.DataFrame:
    
    ruta_fichero = '/opt/covid/Pacientes_Estratificacion.txt'
    archivo_control = '/opt/covid/Ultima_Modificacion_Estratificacion.txt'
    modificado = 1
    # Obtener fecha de última modificación del fichero
    fecha_modificacion = datetime.fromtimestamp(os.path.getmtime(ruta_fichero)).date()

    if os.path.exists(archivo_control):
        with open(archivo_control, 'r') as f:
            linea = f.readline().strip()
            try:
                fecha_guardada = datetime.fromisoformat(linea).date()
                if fecha_modificacion > fecha_guardada:
                    pipeline_logger.info("El archivo ha sido modificado desde la última revisión.")
                else:
                    pipeline_logger.info("El archivo no ha sido modificado desde la última revisión.")
                    modificado = 0
            except ValueError:
                pipeline_logger.error("Formato incorrecto en el archivo de control. Se sobrescribirá.")
               

    with open(archivo_control, 'w') as f:
        f.write(fecha_modificacion.isoformat())
        
    estratificacion = pd.read_csv(ruta_fichero, sep='|', encoding='latin1')
    
    if modificado == 0:
        sys.exit(0)
    else: 
        return estratificacion

@step(enable_cache=False)
def data_preprocessing(estratificacion: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
# Eliminamos los nulos
    estratificacion=estratificacion[estratificacion["ESTRATO"].notnull()]
    estratificacion=estratificacion[estratificacion["COD_POSTAL"].notnull()]
    estratificacion=estratificacion[estratificacion["MUNICIPIO"].notnull()]
    estratificacion = estratificacion[estratificacion["SITUACION"].isin(["CURADO", "FALLECIDO"])]
    estratificacion["OBESITY"] = 0 
    estratificacion["ASTHMA"] = 0
    estratificacion.loc[estratificacion['ETIQUETA'].str.contains("Obesidad", na=False), 'OBESITY'] = 1
    estratificacion.loc[estratificacion['ETIQUETA'].str.contains("Asma", na=False), 'ASTHMA'] = 1
    
    estratificacion_copia = estratificacion.copy()
    # Eliminamos variables inecesarias 
    estratificacion_model = estratificacion.drop(columns=["HOSPITAL_REFERENCIA", "EAP_REFERENCIA", "COD_POSTAL", "MUNICIPIO", "TIPO_TEST_CONFIRMADO","FECHA_ESTRATIFICACON","ESTRATO","GMA"]).copy()
    estratificacion_model['FECHA_DIAGNOSTICO'] = pd.to_datetime(
        estratificacion_model['FECHA_DIAGNOSTICO'],
        errors='coerce',
        dayfirst=True  
    )
    estratificacion_model['mes'] = estratificacion_model['FECHA_DIAGNOSTICO'].dt.month
    estratificacion_model = estratificacion_model.drop(columns=['FECHA_DIAGNOSTICO']).copy()
    estratificacion_model["INGRESO"] = estratificacion_model["INGRESO"].map({"SI": 1, "NO": 0})
    estratificacion_model["SEXO"] = estratificacion_model["SEXO"].map({"H": 1, "M": 0})

    estratificacion_model = estratificacion_model.drop(columns=["ETIQUETA"]).copy()

    return estratificacion_model, estratificacion_copia

@step(enable_cache=False)
def determinarCalidadIPIP(data: pd.DataFrame):
        def prediccion(ensemble, X):
            """
            Predice usando un ensemble (lista de modelos).
            Se calcula el promedio de predicciones de cada modelo.
            """
            preds = [model.predict(X) for model in ensemble]  # lista de arrays
            preds = np.array(preds)  # shape: (n_models, n_samples)
            avg_pred = np.round(np.mean(preds, axis=0)).astype(int)
            return avg_pred

        def prediccion_final(ipip, X):
            """
            ipip: lista de ensembles (lista de listas de modelos)
            Para cada ensemble se calcula su predicción.
            Luego se promedian las predicciones de todos los ensembles.
            """
            all_preds = []
            for ensemble in ipip:
                pred_ensemble = prediccion(ensemble, X)
                all_preds.append(pred_ensemble)
            all_preds = np.array(all_preds)  # shape: (n_ensembles, n_samples)
            final_pred = np.round(np.mean(all_preds, axis=0)).astype(int)
            return final_pred
        
        def prediccion_final_prob(ipip, X):
            """
            Similar a prediccion_final, pero usando probabilidades (predict_proba)
            para cada modelo y promediando.
            """
            all_probs = []
            for ensemble in ipip:
                probs_ensemble = [model.predict_proba(X)[:,1] for model in ensemble] 
                probs_ensemble = np.array(probs_ensemble)
                avg_prob_ensemble = np.mean(probs_ensemble, axis=0)
                all_probs.append(avg_prob_ensemble)
            all_probs = np.array(all_probs)
            final_prob = np.mean(all_probs, axis=0)
            return final_prob

        
        modelos = []
        chunks = dict(tuple(data.groupby('mes')))
        predictions = []
        probs = []
        chunk_labels  = []
        real = []
        for t in range(1,11):
            modelo = joblib.load(f"/opt/covid/resultados_Estratificacion/modelos/IPIP_NSCD.chunk{t}.pmin.0.55.pkl")
            modelos.append(modelo)
            next_chunk = chunks[t].drop(columns=['mes'])
            y_chunk = next_chunk['SITUACION'].map({'CURADO': 1, 'FALLECIDO': 0})
            X_chunk = next_chunk.drop(columns=["SITUACION"])
           
            results_class = prediccion_final(modelos[t-1], X_chunk)
            results_prob = prediccion_final_prob(modelos[t-1], X_chunk)
            predictions.extend(results_class.tolist())
            probs.extend(results_prob.tolist())
            chunk_labels.extend([t] * len(next_chunk))
            real.extend(y_chunk.tolist())
            
        df = pd.DataFrame({
            'real': real,
            'predicciones': predictions,
            'probabilidades': probs, 
            'chunk': chunk_labels
        })
        metricas_por_chunk = df.groupby("chunk").apply(
        lambda g: pd.Series({
            'balanced_accuracy': balanced_accuracy_score(g['real'], g['predicciones'])
        })
        )
        
        balanced_acc_previo_a_train = metricas_por_chunk['balanced_accuracy'].mean()
        if (balanced_acc_previo_a_train >= 0.9): 
            mensaje = f"Balanced accuracy por encima del Umbral: {balanced_acc_previo_a_train:.4f}. No se entrenara."
            print(f"Balanced accuracy por encima del Umbral: {balanced_acc_previo_a_train:.4f}. No se entrenara.")
            pipeline_logger.info(mensaje)
            sys.exit(0) 

@step(enable_cache=False)
def trainer(data: pd.DataFrame, estratificacion_copia: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "final_df"], 
    Annotated[List[str], "models_names"], 
    Annotated[List[List[RandomForestClassifier]], "models"],
    Annotated[pd.DataFrame, "df_10"]
]:
    def prediccion(ensemble, X):
        preds = [model.predict(X) for model in ensemble] 
        preds = np.array(preds) 
        avg_pred = np.round(np.mean(preds, axis=0)).astype(int)
        return avg_pred

    def prediccion_final(ipip, X):
  
        all_preds = []
        for ensemble in ipip:
            pred_ensemble = prediccion(ensemble, X)
            all_preds.append(pred_ensemble)
        all_preds = np.array(all_preds)  
        final_pred = np.round(np.mean(all_preds, axis=0)).astype(int)
        return final_pred

    def prediccion_final_prob(ipip, X):
    
        all_probs = []
        for ensemble in ipip:
            probs_ensemble = [model.predict_proba(X)[:,1] for model in ensemble] 
            probs_ensemble = np.array(probs_ensemble)
            avg_prob_ensemble = np.mean(probs_ensemble, axis=0)
            all_probs.append(avg_prob_ensemble)
        all_probs = np.array(all_probs)
        final_prob = np.mean(all_probs, axis=0)
        return final_prob

    def mt(length):
        return 10 

    def best_models(model, metric_max, metrics, x, y, p):
        model_performance = []
        for ensemble in model:
            pred = prediccion(ensemble, x)
            score = balanced_accuracy_score(y, pred)
            model_performance.append(score)
        sorted_indices = np.argsort(model_performance)[::-1]
        return sorted_indices[:p]
    
    chunks = dict(tuple(data.groupby('mes')))

    chunk_vector = []
    pm_vector = []
    real = []
    pred = []
    pred_previos = []
    real_prob = []
    pred_prob = []
    max_chunk = len(chunks)
    models = []
    models_names = []
    idPaciente_prediccion = {}
  
    prop_mayoritaria = 0.55

    ipip = [] 

    for t in range(1, max_chunk):
        current_chunk = chunks[t].drop(columns=['mes'])

        current_chunk['SITUACION'] = current_chunk['SITUACION'].map({'CURADO': 1, 'FALLECIDO': 0})

        train, test = train_test_split(current_chunk, test_size=0.2, stratify=current_chunk['SITUACION'], random_state=1234)

        discharge = train[train['SITUACION'] == 1]
        expired = train[train['SITUACION'] == 0]

        np_samples = round(len(expired) * 0.75)
        p = int(np.ceil(np.log(0.01) / (np.log(1 - 1 / len(expired)) * np_samples)))
        b = int(np.ceil(np.log(0.01) / (np.log(1 - 1 / np_samples) * np_samples)))

        dfs = []
        for _ in range(p):
            id_expired = expired.sample(n=np_samples, replace=True, random_state=1234)
            id_discharge = discharge.sample(n=round(np_samples * prop_mayoritaria / (1 - prop_mayoritaria)), random_state=1234)
            dfs.append(pd.concat([id_discharge, id_expired]))

        if t == 1:
            E = []
            for df in dfs:
                Ek = []
                i = 0
                while len(Ek) <= b and i < mt(len(Ek)):
                    sample_df = df.sample(frac=1, replace=True, random_state=1234)
                    X_train = sample_df.drop(columns=['SITUACION'])
                    y_train = sample_df['SITUACION']
                    

                    rf = RandomForestClassifier(n_estimators=200, random_state=1234)
                    rf.fit(X_train, y_train)

                    if len(Ek) == 0:
                        metricas_ensemble = -np.inf
                    else:
                        y_pred = prediccion(Ek, test.drop(columns=['SITUACION']))
                        metricas_ensemble = balanced_accuracy_score(test['SITUACION'], y_pred)

                    Ek.append(rf)
                    y_pred_new = prediccion(Ek, test.drop(columns=['SITUACION']))
                    metricas_ensemble_2 = balanced_accuracy_score(test['SITUACION'], y_pred_new)

                    if metricas_ensemble_2 <= metricas_ensemble:
                        i += 1
                        Ek.pop()
                    else:
                        i = 0
                E.append(Ek)
            ipip = E
        else:
            for df in dfs:
                Ek = []
                i = 0
                while len(Ek) <= b and i < mt(len(Ek)):
                    sample_df = df.sample(frac=1, replace=True, random_state=1234)
                    X_train = sample_df.drop(columns=['SITUACION'])
                    y_train = sample_df['SITUACION']
                    rf = RandomForestClassifier(n_estimators=200, random_state=1234)
                    rf.fit(X_train, y_train)

                    if len(Ek) == 0:
                        metricas_ensemble = -np.inf
                    else:
                        y_pred = prediccion(Ek, test.drop(columns=['SITUACION']))
                        metricas_ensemble = balanced_accuracy_score(test['SITUACION'], y_pred)

                    Ek.append(rf)
                    y_pred_new = prediccion(Ek, test.drop(columns=['SITUACION']))
                    metricas_ensemble_2 = balanced_accuracy_score(test['SITUACION'], y_pred_new)

                    if metricas_ensemble_2 <= metricas_ensemble:
                        i += 1
                        Ek.pop()
                    else:
                        i = 0
                ipip.append(Ek)
            ipip = [ipip[i] for i in best_models(ipip, "BAL_ACC", ["BAL_ACC"], test.drop(columns=['SITUACION']), test['SITUACION'], p)]
        
        modelos_previos = []
        for number in range(1, max_chunk):
            modelos_previos.append(joblib.load(f"/opt/covid/resultados_Estratificacion/modelos/IPIP_NSCD.chunk{number}.pmin.0.55.pkl"))
            
        next_chunk = chunks[t + 1].drop(columns=['mes'])
        next_chunk['SITUACION'] = next_chunk['SITUACION'].map({'CURADO': 1, 'FALLECIDO': 0})
        X_next = next_chunk.drop(columns=['SITUACION'])
        y_next = next_chunk['SITUACION']
        
        results_class = prediccion_final(ipip, X_next)
        results_class_previos = prediccion_final(modelos_previos[t-1], X_next)
        results_prob = prediccion_final_prob(ipip, X_next)
        
        for id_paciente, prediccion_class in zip(X_next["ID_PACIENTE"], results_class):
         idPaciente_prediccion[id_paciente] = prediccion_class
    

        df_filtrado = estratificacion_copia[estratificacion_copia['ID_PACIENTE'].isin(idPaciente_prediccion.keys())]

 
        df_10 = df_filtrado.sample(10).copy()

   
        df_10['prediccion'] = df_10['ID_PACIENTE'].map(idPaciente_prediccion)

        
        real.extend(y_next.tolist())
        pred.extend(results_class.tolist())
        pred_previos.extend(results_class_previos.tolist())
        pred_prob.extend(results_prob.tolist())
        real_prob.extend(y_next.tolist())
        chunk_vector.extend([t] * len(y_next))
        pm_vector.extend([0.55] * len(y_next))
        
        models.extend(ipip)
        models_names.append(f"/opt/covid/resultados_Estratificacion/modelos/IPIP_NSCD.chunk{t}.pmin.0.55.pkl")
        
    final_df = pd.DataFrame({
        'real': real,
        'pred': pred,
        'pred_previos': pred_previos,
        'pred_prob': pred_prob,
        'real_prob': real_prob,
        'p_min': pm_vector,
        'chunk': chunk_vector
    })
    final_df.to_pickle('/opt/covid/resultados_Estratificacion/results_IPIP_NSCD.pkl')
    return final_df, models_names, models, df_10

@step(enable_cache=False)
def evaluacion(final_df: pd.DataFrame) -> Tuple[float,float]:
    y_true = final_df['real']
    y_pred = final_df['pred']

    
    chunk_metrics = final_df.groupby("chunk").apply(
    lambda g: pd.Series({
        'balanced_accuracy': balanced_accuracy_score(g['real'], g['pred'])
    })
    )
    
    balanced_acc = chunk_metrics['balanced_accuracy'].mean()
    
    chunk_metrics_previos = final_df.groupby("chunk").apply(
    lambda g: pd.Series({
        'balanced_accuracy_previo': balanced_accuracy_score(g['real'], g['pred_previos'])
    })
    )
    balanced_acc_previo = chunk_metrics_previos['balanced_accuracy_previo'].mean()
 
    # Guardar en archivo
    with open("/opt/covid/resultados_Estratificacion/modelos/metricas_totales.txt", "w") as f:
        f.write(f"Balanced Accuracy Promedio de los Chunks: {balanced_acc:.4f}\n")
        f.write("Primeros 5 valores reales\n")
        f.write(y_true.head().to_string())
        f.write("\n")
        f.write("Primeros 5 valores predichos\n")
        f.write(y_pred.head().to_string())

    # Analizar métricas por chunk
    chunk_metrics = []
    for chunk, g in final_df.groupby("chunk"):
        y_true_chunk = g['real'].values
        y_pred_chunk = g['pred'].values
        
        # Calcular métrica puntual
        bal_acc = balanced_accuracy_score(y_true_chunk, y_pred_chunk)
        
        # Bootstrapping para intervalo de confianza
        n_bootstrap = 1000
        rng = np.random.RandomState(1234)
        bal_acc_samples = []
        for _ in range(n_bootstrap):
            indices = rng.choice(len(y_true_chunk), len(y_true_chunk), replace=True)
            bal_acc_sample = balanced_accuracy_score(y_true_chunk[indices], y_pred_chunk[indices])
            bal_acc_samples.append(bal_acc_sample)
        lower = np.percentile(bal_acc_samples, 2.5)
        upper = np.percentile(bal_acc_samples, 97.5)
        
        chunk_metrics.append({
            "chunk": chunk,
            "balanced_accuracy": bal_acc,
            "balanced_accuracy_lower": lower,
            "balanced_accuracy_upper": upper
        })

    chunk_metrics = pd.DataFrame(chunk_metrics).set_index("chunk")

    # Configuración de estilo bonito con seaborn
    sns.set(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Curva principal
    ax.plot(
        chunk_metrics.index,
        chunk_metrics["balanced_accuracy"],
        label="Balanced Accuracy",
        color="royalblue",
        marker="o",
        linewidth=2
    )

    # Banda del intervalo de confianza
    ax.fill_between(
        chunk_metrics.index,
        chunk_metrics["balanced_accuracy_lower"],
        chunk_metrics["balanced_accuracy_upper"],
        color="royalblue",
        alpha=0.2,
        label="IC 95%"
    )

    # Títulos y etiquetas
    ax.set_title("Evolución del desempeño por chunk (mes) con IC 95%", fontsize=18, weight="bold")
    ax.set_xlabel("Chunk", fontsize=14)
    ax.set_ylabel("Balanced Accuracy", fontsize=14)

    # Leyenda elegante
    ax.legend(loc="upper left", fontsize=12, frameon=True)

    # Opcional: rotar etiquetas del eje x si son fechas u ocupan mucho espacio
    plt.xticks(rotation=45)

    # Ajustar layout para que no se corte nada
    plt.tight_layout()

    # Guardar como imagen
    plt.savefig("/opt/covid/resultados_Estratificacion/metricas_por_chunk.png", dpi=150)


    return balanced_acc, balanced_acc_previo

@step(enable_cache=False)
def save_model(balanced_accuracy: float, balanced_accuracy_previo:float, models_names: List[str], models: List[List[RandomForestClassifier]], df_10: pd.DataFrame ): 
             if(balanced_accuracy < 0.9 or balanced_accuracy < balanced_accuracy_previo):
                 error_msg = f"Balanced accuracy demasiado bajo: {balanced_accuracy:.4f}. No se guardarán los modelos."
                 pipeline_logger.error(error_msg, exc_info=True)
                 raise ValueError(error_msg)
             else: 
                 for i in range(len(models_names)):
                    joblib.dump(models[i], models_names[i]) 
                    df_10.to_csv("/opt/covid/resultados_Estratificacion/predicciones_ids.csv", index=False)
                
                  

@pipeline
def pipeline():
    df = data_loader()
    estratificacion_model, estratificacion_copia = data_preprocessing(df)
    determinarCalidadIPIP(estratificacion_model)
    final_df, models_names, models, df_10 = trainer(estratificacion_model, estratificacion_copia)
    balanced_accuracy, balanced_accuracy_previo = evaluacion(final_df)
    save_model(balanced_accuracy, balanced_accuracy_previo, models_names, models, df_10)
    


if __name__ == "__main__":
    pipeline()