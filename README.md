
# 🍼 Newborn Health Risk Prediction using Machine Learning & Deep Learning

This project predicts newborn health risk levels using clinical features such as **gestational age**, **birth weight**, **APGAR score**, **jaundice level**, and vital parameters. Multiple **Machine Learning models** and a **Deep Learning Neural Network (ANN)** were developed, trained, tuned, and compared to identify the best-performing approach for early neonatal risk assessment.

---

## 🚀 Project Overview

Early identification of newborn health risks can support pediatricians and neonatologists in taking timely medical intervention.
This system uses supervised learning to classify newborns as **Healthy** or **At-Risk**.

The workflow includes:

* Data preprocessing & feature engineering
* Training multiple ML classifiers
* Hyperparameter tuning (GridSearchCV)
* Neural Network with SMOTE oversampling & Dropout + BatchNorm
* Performance evaluation & comparison

---

## 📂 Dataset Description

The dataset contains clinical attributes of newborns, including:

| Feature         | Description                         |
| --------------- | ----------------------------------- |
| Gestational Age | Weeks of pregnancy                  |
| Birth Weight    | Weight of newborn in grams          |
| APGAR Score     | Key newborn health indicator        |
| Jaundice Level  | Clinical bilirubin evaluation       |
| Vital Signs     | Heart rate, respiration etc.        |
| Risk Level      | Target variable (Healthy / At-Risk) |

> Personal identifiers were removed; only medical features were used.

---

## 🧠 Models Implemented

### ✅ Machine Learning Models

* Logistic Regression
* Decision Tree
* Random Forest *(Best performing model)*
* Support Vector Machine
* k-Nearest Neighbors
* Naive Bayes

### 🤖 Deep Learning Model

* Artificial Neural Network (ANN)

  * SMOTE oversampling for imbalance handling
  * Batch Normalization + Dropout
  * Standard scaling & label encoding

---

## ⚙️ Model Tuning

* **Random Forest** tuned using GridSearchCV
* **Neural Network** improved with:

  * SMOTE oversampling
  * BatchNormalization layers
  * Dropout regularization

---

## 📊 Results

| Model                      | Accuracy   |
| -------------------------- | ---------- |
| Random Forest (Tuned)      | ✅ **~99%** |
| Neural Network (Optimized) | ✅ **~95%** |
| KNN                        | ~92%       |

### Key findings:

* Random Forest provided the most reliable predictions
* ANN performed strongly after balancing and regularization
* Gestational age, birth weight, APGAR score, and jaundice level were most influential features

---

## 🧪 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Cross-validation

---

## 📦 Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-Learn**
* **TensorFlow / Keras**
* **Matplotlib / Seaborn**
* **Imbalanced-Learn (SMOTE)**

---

## 🏗️ Project Structure

```
📁 newborn-risk-prediction
 ├── 📄 README.md
 ├── 📓 notebook.ipynb
 ├── 📄 Dataset
 
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python notebook.ipynb
```

For Streamlit UI (optional):

```bash
streamlit run app.py
```

---

## 🎯 Conclusion

This system demonstrates strong performance in early newborn health risk detection.
The **tuned Random Forest model** showed the highest reliability, followed by the optimized ANN, making it useful for **clinical decision support** and **healthcare screening systems**.

---

## 🚀 Future Enhancements

* Real-time prediction dashboard (Streamlit)
* Hospital EHR integration
* Explainable AI (SHAP/LIME)
* Deployment as API / mobile app

---

## 🤝 Contributions

Pull requests and suggestions are welcome!

---

## 📧 Contact

Author: **Teja**
For queries, feel free to connect via GitHub or LinkedIn 🤝

---
