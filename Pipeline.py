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
import matplotlib.pyplot as plt
from datetime import datetime
import sys

@step(enable_cache=False)
def data_loader() -> pd.DataFrame:
    
    ruta_fichero = '/opt/covid/Pacientes_Estratificacion.txt'
    archivo_control = '/opt/covid/Ultima_Modificacion_Estratificacion.txt'
    modificado = 1
    
    fecha_modificacion = datetime.fromtimestamp(os.path.getmtime(ruta_fichero)).date()

    if os.path.exists(archivo_control):
        with open(archivo_control, 'r') as f:
            linea = f.readline().strip()
            try:
                fecha_guardada = datetime.fromisoformat(linea).date()
                if fecha_modificacion > fecha_guardada:
                    print("El archivo ha sido modificado desde la última revisión.")
                else:
                    print("El archivo no ha sido modificado desde la última revisión.")
                    modificado = 0
            except ValueError:
                print("Formato incorrecto en el archivo de control. Se sobrescribirá.")
               

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
def trainer(data: pd.DataFrame, estratificacion_copia: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "final_df"], 
    Annotated[List[str], "models_names"], 
    Annotated[List[List[RandomForestClassifier]], "models"],
    Annotated[pd.DataFrame, "df_10"]
]:
    def prediccion(ensemble, X):
        """
        Predice usando un ensemble (lista de modelos).
        Se calcula el promedio de predicciones de cada modelo.
        """
        preds = [model.predict(X) for model in ensemble]  
        preds = np.array(preds) 
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
        all_preds = np.array(all_preds)  
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
            modelos_previos.extend(f"/opt/covid/resultados_Estratificacion/modelos/IPIP_NSCD.chunk{number}.pmin.0.55.pkl")
            
        next_chunk = chunks[t + 1].drop(columns=['mes'])
        next_chunk['SITUACION'] = next_chunk['SITUACION'].map({'CURADO': 1, 'FALLECIDO': 0})
        X_next = next_chunk.drop(columns=['SITUACION'])
        y_next = next_chunk['SITUACION']
        
        results_class = prediccion_final(ipip, X_next)
        results_class_previos = prediccion_final(modelos_previos[t], X_next)
        results_prob = prediccion_final_prob(ipip, X_next)
        
        for id_paciente, prediccion_class in zip(X_next["ID_PACIENTE"], results_class):
         idPaciente_prediccion[id_paciente] = prediccion_class
    

        df_filtrado = estratificacion_copia[estratificacion_copia['ID_PACIENTE'].isin(idPaciente_prediccion.keys())]

 
        df_10 = df_filtrado.sample (10).copy()

   
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
def evaluacion(final_df: pd.DataFrame) -> float:
    y_true = final_df['real']
    y_pred = final_df['pred']
    y_prob = final_df['pred_prob']
    y_pred_previos = []
    if 'pred_previos' in final_df.columns:
        y_pred_previos = final_df['pred_previos']
     
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    if y_pred_previos:
        balanced_acc_previo = balanced_accuracy_score(y_true, y_pred_previos)
    else: 
        balanced_acc_previo = 0.0


    with open("/opt/covid/resultados_Estratificacion/modelos/metricas_totales.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write("Primeros 5 valores reales\n")
        f.write(y_true.head().to_string())
        f.write("\n")
        f.write("Primeros 5 valores predichos\n")
        f.write(y_pred.head().to_string())


    chunk_metrics = final_df.groupby("chunk").apply(
        lambda g: pd.Series({
            'accuracy': accuracy_score(g['real'], g['pred']),
            'precision': precision_score(g['real'], g['pred']),
            'recall': recall_score(g['real'], g['pred']),
            'f1': f1_score(g['real'], g['pred']),
            'balanced_accuracy': balanced_accuracy_score(g['real'], g['pred'])
        
        })
    )

    plt.figure(figsize=(10, 6))
    chunk_metrics[['balanced_accuracy']].plot(marker='o')
    plt.title("Evolución del desempeño por chunk (mes)")
    plt.xlabel("Chunk")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/opt/covid/resultados_Estratificacion/metricas_por_chunk.png")  
    plt.close() 

    cm = confusion_matrix(final_df['real'], final_df['pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FALLECIDO", "CURADO"])

    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Matriz de Confusión - Total")
    plt.tight_layout()
    plt.savefig("/opt/covid/resultados_Estratificacion/matriz_confusion_total.png")
    plt.close()
    return balanced_acc, balanced_acc_previo

@step(enable_cache=False)
def save_model(balanced_accuracy: float, balanced_accuracy_previo: float, models_names: List[str], models: List[List[RandomForestClassifier]], df_10: pd.DataFrame ): 
             if(balanced_accuracy < 0.9 or balanced_accuracy < balanced_accuracy_previo):
                print("La lista de ensembles no supera el Threshold u obtienen un peor rendimiento que los modelos previos, por lo que no guardaremos el modelo")
             else: 
                 for i in range(len(models_names)):
                    joblib.dump(models[i], models_names[i]) 
                    df_10.to_csv("/opt/covid/resultados_Estratificacion/predicciones_ids.csv", index=False)
                
                  

@pipeline
def pipeline():
    df = data_loader()
    estratificacion_model, estratificacion_copia = data_preprocessing(df)
    final_df, models_names, models, df_10 = trainer(estratificacion_model, estratificacion_copia)
    balanced_accuracy, balanced_accuracy_previo = evaluacion(final_df)
    save_model(balanced_accuracy, balanced_accuracy_previo, models_names, models, df_10)
    


if __name__ == "__main__":
    pipeline()