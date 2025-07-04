# 🦠 COVID-19 Predictive Pipeline with Concept Drift Adaptation

An MLOps pipeline developed with **ZenML** to continuously train and update a predictive model on COVID patient data. The goal is to automatically detect and adapt to **concept drift** in the stratification data provided by the SMS, ensuring that deployed models remain up-to-date with excellent performance while properly notifying the team in case of any issues.

---

## 📌 Introduction

**Concept drift** refers to changes in the relationship between input variables (X) and the target variable (y) over time, causing a previously trained model to lose accuracy.

Our pipeline detects drift **implicitly** by evaluating performance monthly on new data (monthly chunks) and comparing it to the performance of previous models. If performance drops (e.g., balanced accuracy decreases), the pipeline avoids updating the model, ensuring retraining only occurs when the new model actually improves or maintains performance.

---

## 💡 Main Idea

1. **Performance monitoring:**  
   Evaluates performance on the most recent chunks compared to previous models using metrics like *balanced accuracy*.

2. **Adaptive retraining:**  
   If the data distribution changes, the pipeline trains updated ensembles with the latest data, adjusting the model to new patterns.

3. **Automation with CRON:**  
   Automated execution with `crontab` ensures a continuous learning and updating system.

---

## 🔄 Workflow

### 1️⃣ Data Ingestion

- Loads the data and checks the file’s last modification date.
- If no changes are detected since the last run (using a control file), the pipeline exits to optimize resources.
- If changes are detected, the updated dataframe is returned.

### 2️⃣ Data Preprocessing

- Removes nulls in key columns.
- Generates binary variables (e.g., `OBESITY`, `ASTHMA`) from the `ETIQUETA` column.
- Processes the diagnosis date to extract the month (`mes`) and segment the data into **monthly chunks**.
- Converts categorical variables like `INGRESO` and `SEXO` to numeric format.

### 3️⃣ Training

- Splits data into monthly chunks, treating them as independent mini-datasets.
- Creates multiple balanced bootstraps using the **IPIP** (Iterative Proportional Importance Pruning) strategy.
- Each bootstrap trains a Random Forest ensemble, retaining only models that improve the ensemble’s performance on a validation set.
- Generates predictions on the following chunk, simulating future data predictions.
- Saves results and metric comparisons with previous models.

### 4️⃣ Validation

- Calculates global and per-chunk metrics: *accuracy*, *precision*, *recall*, *f1*, and *balanced accuracy*.
- Generates time series plots of balanced accuracy evolution and an accumulated confusion matrix.
- Saves metrics and charts as monitoring evidence.

### 5️⃣ Model Deployment

- Compares the performance of the new model with the previous one.
- Only saves the new model if:
  - Its performance exceeds the expected minimum threshold.
  - The model improves or maintains performance compared to the previous model.
- Saves prediction examples for auditing and traceability.

---

## 🔍 Concept Drift

The approach relies on segmenting data into monthly chunks to quickly detect and respond to behavioral changes. For each new chunk, the performance of the current model is compared with previous models, effectively functioning as a supervised drift test.

The pipeline avoids overwriting the model if the new one does not show significant improvement, preventing degradation due to temporary drift or noise. The **IPIP** methodology further improves handling of imbalanced datasets.

---

## 🧩 IPIP (Iterative Proportional Importance Pruning)

The **IPIP** technique combines bagging, ensembles, and iterative balancing to improve predictions on problems with class imbalance and time series. It allows:

- Balancing classes in each bootstrap.
- Selecting models that truly improve the ensemble.
- Dynamically adapting the model to changes in data distribution (concept drift).

---

## 🚀 Automation

Integrates scheduled tasks with `crontab` to periodically run the pipeline, ensuring a continuous learning system with no manual intervention required.

---

## 🗂️ Technologies

- ZenML
- Python 3.10  
- Scikit-learn  
- Pandas  
- Matplotlib / Seaborn  
- CRON for automation

---

