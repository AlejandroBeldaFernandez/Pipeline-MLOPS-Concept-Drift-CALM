from zenml import pipeline, step
from typing import Annotated, Tuple, List
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import logging  
from pathlib import Path


def ensemble_prediction(ensemble, X):
    """Realiza predicción usando conjunto de modelos"""
    preds = [model.predict(X) for model in ensemble] 
    preds = np.array(preds) 
    avg_pred = np.round(np.mean(preds, axis=0)).astype(int)
    return avg_pred

def final_prediction(model_collection, X):
    """Predicción final usando múltiples ensembles"""
    all_preds = []
    for ensemble in model_collection:
        pred_ensemble = ensemble_prediction(ensemble, X)
        all_preds.append(pred_ensemble)
    all_preds = np.array(all_preds)  
    final_pred = np.round(np.mean(all_preds, axis=0)).astype(int)
    return final_pred

def final_prediction_prob(model_collection, X):
    """Predicción de probabilidades usando múltiples ensembles"""
    all_probs = []
    for ensemble in model_collection:
        probs_ensemble = [model.predict_proba(X)[:,1] for model in ensemble] 
        probs_ensemble = np.array(probs_ensemble)
        avg_prob_ensemble = np.mean(probs_ensemble, axis=0)
        all_probs.append(avg_prob_ensemble)
    all_probs = np.array(all_probs)
    final_prob = np.mean(all_probs, axis=0)
    return final_prob

def max_iterations(length):
    """Calcula máximo de iteraciones"""
    return 10 

def select_best_models(model_list, metric_max, metrics, x, y, p):
    """Selecciona los mejores modelos basado en métricas"""
    model_performance = []
    for ensemble in model_list:
        pred = ensemble_prediction(ensemble, x)
        score = balanced_accuracy_score(y, pred)
        model_performance.append(score)
    sorted_indices = np.argsort(model_performance)[::-1]
    return sorted_indices[:p]


def setup_pipeline_logging():
    """Configura el sistema de logging para la pipeline"""
    # Crear directorio de logs
    log_dir = Path('/path/to/logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configurar formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Logger principal de la pipeline
    logger = logging.getLogger('ml_pipeline')
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
    """Carga datos desde archivo y verifica modificaciones"""
    data_file = '/path/to/data/dataset.txt'
    control_file = '/path/to/control/last_modified.txt'
    is_modified = 1
    file_modified_date = datetime.fromtimestamp(os.path.getmtime(data_file)).date()

    if os.path.exists(control_file):
        with open(control_file, 'r') as f:
            line = f.readline().strip()
            try:
                saved_date = datetime.fromisoformat(line).date()
                if file_modified_date > saved_date:
                    pipeline_logger.info("El archivo ha sido modificado desde la última revisión.")
                else:
                    pipeline_logger.info("El archivo no ha sido modificado desde la última revisión.")
                    is_modified = 0
            except ValueError:
                pipeline_logger.error("Formato incorrecto en el archivo de control. Se sobrescribirá.")
               
    with open(control_file, 'w') as f:
        f.write(file_modified_date.isoformat())
        
    dataset = pd.read_csv(data_file, sep='|', encoding='latin1')
    
    if is_modified == 0:
        sys.exit(0)
    else: 
        return dataset

@step(enable_cache=False)
def data_preprocessing(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocesa los datos para el modelo"""
    # Filtrar registros válidos
    dataset = dataset[dataset["category_field"].notnull()]
    dataset = dataset[dataset["postal_code"].notnull()]
    dataset = dataset[dataset["location"].notnull()]
    dataset = dataset[dataset["status"].isin(["outcome_a", "outcome_b"])]
    
    # Crear variables derivadas
    dataset["feature_1"] = 0 
    dataset["feature_2"] = 0
    dataset.loc[dataset['tags'].str.contains("condition_1", na=False), 'feature_1'] = 1
    dataset.loc[dataset['tags'].str.contains("condition_2", na=False), 'feature_2'] = 1
    
    dataset_copy = dataset.copy()
    
    # Eliminar variables innecesarias 
    model_data = dataset.drop(columns=["ref_hospital", "ref_center", "postal_code", "location", "test_type", "date_category", "category_field", "group_field"]).copy()
    
    # Procesamiento de fechas
    model_data['date_diagnosis'] = pd.to_datetime(
        model_data['date_diagnosis'],
        errors='coerce',
        dayfirst=True  
    )
    model_data['time_period'] = model_data['date_diagnosis'].dt.month
    model_data = model_data.drop(columns=['date_diagnosis']).copy()
    
    # Codificación de variables categóricas
    model_data["hospitalization"] = model_data["hospitalization"].map({"yes": 1, "no": 0})
    model_data["gender"] = model_data["gender"].map({"male": 1, "female": 0})
    model_data['status'] = model_data['status'].map({'outcome_a': 1, 'outcome_b': 0})
    
    model_data = model_data.drop(columns=["tags"]).copy()

    return model_data, dataset_copy

@step(enable_cache=False)
def evaluate_existing_models(data: pd.DataFrame):
    """Evalúa modelos existentes antes de entrenar nuevos"""
    models = []
    chunks = dict(tuple(data.groupby('time_period')))
    predictions = []
    probs = []
    chunk_labels = []
    real = []
    
    for t in range(1, 11):
        model = joblib.load(f"/path/to/models/model.chunk{t}.threshold.0.55.pkl")
        models.append(model)
        next_chunk = chunks[t].drop(columns=['time_period'])
        y_chunk = next_chunk['status']
        X_chunk = next_chunk.drop(columns=["status"])
       
        results_class = final_prediction(models[t-1], X_chunk)
        results_prob = final_prediction_prob(models[t-1], X_chunk)
        predictions.extend(results_class.tolist())
        probs.extend(results_prob.tolist())
        chunk_labels.extend([t] * len(next_chunk))
        real.extend(y_chunk.tolist())
        
    df = pd.DataFrame({
        'real': real,
        'predictions': predictions,
        'probabilities': probs, 
        'chunk': chunk_labels
    })
    
    metrics_by_chunk = df.groupby("chunk").apply(
        lambda g: pd.Series({
            'balanced_accuracy': balanced_accuracy_score(g['real'], g['predictions'])
        })
    )
    
    prev_balanced_acc = metrics_by_chunk['balanced_accuracy'].mean()
    if prev_balanced_acc >= 0.97: 
        message = f"Balanced accuracy por encima del umbral: {prev_balanced_acc:.4f}. No se entrenará."
        print(message)
        pipeline_logger.info(message)
        sys.exit(0) 

@step(enable_cache=False)
def trainer(data: pd.DataFrame, dataset_copy: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "final_df"], 
    Annotated[List[str], "models_names"], 
    Annotated[List[List[RandomForestClassifier]], "models"],
    Annotated[pd.DataFrame, "sample_df"]
]:
    """Entrena los modelos de machine learning"""
    chunks = dict(tuple(data.groupby('time_period')))

    chunk_vector = []
    threshold_vector = []
    real = []
    pred = []
    pred_previous = []
    real_prob = []
    pred_prob = []
    max_chunk = len(chunks)
    models = []
    models_names = []
    patient_prediction = {}
  
    majority_prop = 0.55
    model_collection = [] 

    for t in range(1, max_chunk):
        current_chunk = chunks[t].drop(columns=['time_period'])

        train, test = train_test_split(current_chunk, test_size=0.2, stratify=current_chunk['status'], random_state=1234)

        positive_class = train[train['status'] == 1]
        negative_class = train[train['status'] == 0]

        np_samples = round(len(negative_class) * 0.75)
        p = int(np.ceil(np.log(0.01) / (np.log(1 - 1 / len(negative_class)) * np_samples)))
        b = int(np.ceil(np.log(0.01) / (np.log(1 - 1 / np_samples) * np_samples)))

        dfs = []
        for _ in range(p):
            id_negative = negative_class.sample(n=np_samples, replace=True, random_state=1234)
            id_positive = positive_class.sample(n=round(np_samples * majority_prop / (1 - majority_prop)), random_state=1234)
            dfs.append(pd.concat([id_positive, id_negative]))

        if t == 1:
            E = []
            for df in dfs:
                Ek = []
                i = 0
                while len(Ek) <= b and i < max_iterations(len(Ek)):
                    sample_df = df.sample(frac=1, replace=True, random_state=1234)
                    X_train = sample_df.drop(columns=['status'])
                    y_train = sample_df['status']
                    
                    rf = RandomForestClassifier(n_estimators=200, random_state=1234)
                    rf.fit(X_train, y_train)

                    if len(Ek) == 0:
                        ensemble_metrics = -np.inf
                    else:
                        y_pred = ensemble_prediction(Ek, test.drop(columns=['status']))
                        ensemble_metrics = balanced_accuracy_score(test['status'], y_pred)

                    Ek.append(rf)
                    y_pred_new = ensemble_prediction(Ek, test.drop(columns=['status']))
                    ensemble_metrics_2 = balanced_accuracy_score(test['status'], y_pred_new)

                    if ensemble_metrics_2 <= ensemble_metrics:
                        i += 1
                        Ek.pop()
                    else:
                        i = 0
                E.append(Ek)
            model_collection = E
        else:
            for df in dfs:
                Ek = []
                i = 0
                while len(Ek) <= b and i < max_iterations(len(Ek)):
                    sample_df = df.sample(frac=1, replace=True, random_state=1234)
                    X_train = sample_df.drop(columns=['status'])
                    y_train = sample_df['status']
                    rf = RandomForestClassifier(n_estimators=200, random_state=1234)
                    rf.fit(X_train, y_train)

                    if len(Ek) == 0:
                        ensemble_metrics = -np.inf
                    else:
                        y_pred = ensemble_prediction(Ek, test.drop(columns=['status']))
                        ensemble_metrics = balanced_accuracy_score(test['status'], y_pred)

                    Ek.append(rf)
                    y_pred_new = ensemble_prediction(Ek, test.drop(columns=['status']))
                    ensemble_metrics_2 = balanced_accuracy_score(test['status'], y_pred_new)

                    if ensemble_metrics_2 <= ensemble_metrics:
                        i += 1
                        Ek.pop()
                    else:
                        i = 0
                model_collection.append(Ek)
            model_collection = [model_collection[i] for i in select_best_models(model_collection, "BAL_ACC", ["BAL_ACC"], test.drop(columns=['status']), test['status'], p)]
        
        previous_models = []
        for number in range(1, max_chunk):
            previous_models.append(joblib.load(f"/path/to/models/model.chunk{number}.threshold.0.55.pkl"))
            
        next_chunk = chunks[t + 1].drop(columns=['time_period'])
        next_chunk['status'] = next_chunk['status']
        X_next = next_chunk.drop(columns=['status'])
        y_next = next_chunk['status']
        
        results_class = final_prediction(model_collection, X_next)
        results_class_previous = final_prediction(previous_models[t-1], X_next)
        results_prob = final_prediction_prob(model_collection, X_next)
        
        for patient_id, prediction_class in zip(X_next["patient_id"], results_class):
            patient_prediction[patient_id] = prediction_class
    
        filtered_df = dataset_copy[dataset_copy['patient_id'].isin(patient_prediction.keys())]
        sample_df = filtered_df.sample(10).copy()
        sample_df['prediction'] = sample_df['patient_id'].map(patient_prediction)
        
        real.extend(y_next.tolist())
        pred.extend(results_class.tolist())
        pred_previous.extend(results_class_previous.tolist())
        pred_prob.extend(results_prob.tolist())
        real_prob.extend(y_next.tolist())
        chunk_vector.extend([t] * len(y_next))
        threshold_vector.extend([0.55] * len(y_next))
        
        models.extend(model_collection)
        models_names.append(f"/path/to/models/model.chunk{t}.threshold.0.55.pkl")
        
    final_df = pd.DataFrame({
        'real': real,
        'pred': pred,
        'pred_previous': pred_previous,
        'pred_prob': pred_prob,
        'real_prob': real_prob,
        'threshold': threshold_vector,
        'chunk': chunk_vector
    })
    final_df.to_pickle('/path/to/results/results.pkl')
    return final_df, models_names, models, sample_df

@step(enable_cache=False)
def evaluation(final_df: pd.DataFrame) -> Tuple[float, float]:
    """Evalúa el rendimiento de los modelos"""
    y_true = final_df['real']
    y_pred = final_df['pred']
    
    chunk_metrics = final_df.groupby("chunk").apply(
        lambda g: pd.Series({
            'balanced_accuracy': balanced_accuracy_score(g['real'], g['pred'])
        })
    )
    
    balanced_acc = chunk_metrics['balanced_accuracy'].mean()
    
    chunk_metrics_previous = final_df.groupby("chunk").apply(
        lambda g: pd.Series({
            'balanced_accuracy_previous': balanced_accuracy_score(g['real'], g['pred_previous'])
        })
    )
    balanced_acc_previous = chunk_metrics_previous['balanced_accuracy_previous'].mean()
 
    # Guardar métricas en archivo
    with open("/path/to/results/total_metrics.txt", "w") as f:
        f.write(f"Balanced Accuracy Promedio: {balanced_acc:.4f}\n")
        f.write("Primeros 5 valores reales\n")
        f.write(y_true.head().to_string())
        f.write("\n")
        f.write("Primeros 5 valores predichos\n")
        f.write(y_pred.head().to_string())

    # Analizar métricas por chunk con bootstrapping
    chunk_metrics_list = []
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
        
        chunk_metrics_list.append({
            "chunk": chunk,
            "balanced_accuracy": bal_acc,
            "balanced_accuracy_lower": lower,
            "balanced_accuracy_upper": upper
        })

    chunk_metrics_df = pd.DataFrame(chunk_metrics_list).set_index("chunk")
    chunk_metrics_df.to_pickle('/path/to/results/chunk_results.pkl')
    
    # Crear gráfico de métricas
    sns.set(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 7))

    # Curva principal
    ax.plot(
        chunk_metrics_df.index,
        chunk_metrics_df["balanced_accuracy"],
        label="Balanced Accuracy",
        color="royalblue",
        marker="o",
        linewidth=2
    )

    # Banda del intervalo de confianza
    ax.fill_between(
        chunk_metrics_df.index,
        chunk_metrics_df["balanced_accuracy_lower"],
        chunk_metrics_df["balanced_accuracy_upper"],
        color="royalblue",
        alpha=0.2,
        label="IC 95%"
    )

    # Títulos y etiquetas
    ax.set_title("Evolución del desempeño por chunk con IC 95%", fontsize=18, weight="bold")
    ax.set_xlabel("Chunk", fontsize=14)
    ax.set_ylabel("Balanced Accuracy", fontsize=14)

    # Leyenda
    ax.legend(loc="lower right", fontsize=12, frameon=True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("/path/to/results/metrics_by_chunk.png", dpi=150)

    return balanced_acc, balanced_acc_previous

@step(enable_cache=False)
def save_model(balanced_accuracy: float, balanced_accuracy_previous: float, 
               models_names: List[str], models: List[List[RandomForestClassifier]], 
               sample_df: pd.DataFrame): 
    """Guarda los modelos si cumplen con los criterios de calidad"""
    if balanced_accuracy < 0.9 or balanced_accuracy < balanced_accuracy_previous:
        error_msg = f"Balanced accuracy demasiado bajo: {balanced_accuracy:.4f}. No se guardarán los modelos."
        pipeline_logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)
    else: 
        for i in range(len(models_names)):
            joblib.dump(models[i], models_names[i]) 
        sample_df.to_csv("/path/to/results/sample_predictions.csv", index=False)
                                 

@pipeline
def ml_pipeline():
    """Pipeline principal de machine learning"""
    df = data_loader()
    model_data, dataset_copy = data_preprocessing(df)
    evaluate_existing_models(model_data)
    final_df, models_names, models, sample_df = trainer(model_data, dataset_copy)
    balanced_accuracy, balanced_accuracy_previous = evaluation(final_df)
    save_model(balanced_accuracy, balanced_accuracy_previous, models_names, models, sample_df)
    

if __name__ == "__main__":
    ml_pipeline()
