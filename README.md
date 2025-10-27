# 🧠 Fake Job Posting Detection using NLP & Machine Learning

## 📘 Project Overview
The **Fake Job Posting Detection** project aims to classify job postings as *real* or *fake* using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques.  
Two different datasets were used to compare model performance and ensure robustness:
1. **Fake Job Posting Dataset**
2. **Job Train Dataset**

Each dataset underwent text preprocessing, feature extraction using **TF-IDF**, and training using multiple ML algorithms including **Naive Bayes**, **Logistic Regression**, **SVM**, and **Artificial Neural Network (ANN - Keras)**.

---

## 🎯 Objectives
- Detect fraudulent job postings automatically using text analytics.  
- Apply **NLP preprocessing** techniques to clean and vectorize text data.  
- Train and compare multiple ML models.  
- Evaluate performance using metrics like **Accuracy, Precision, Recall, F1-score, and AUC**.  
- Identify the best-performing model for both datasets.

---

## 🧩 Datasets Used

### 1️⃣ Fake Job Posting Dataset
- Source: Kaggle  
- Contains job advertisements with detailed descriptions, titles, company info, etc.  
- Target Column → `fraudulent` (0 = Real, 1 = Fake)  
- Includes both real and fake job listings scraped from online portals.

### 2️⃣ Job Train Dataset
- Used for performance comparison and validation.  
- Contains labeled job-related data with text and categorical features.  
- Target column represents whether the job post is **real** or **fake**.

---

## ⚙️ Technologies & Libraries Used
- **Python 3.x**
- **Pandas**, **NumPy** – Data handling and preprocessing  
- **Matplotlib**, **Seaborn** – Data visualization  
- **Scikit-learn** – ML model training and evaluation  
- **TensorFlow / Keras** – ANN model building  
- **NLTK** – NLP text preprocessing  
- **TF-IDF Vectorizer** – Text feature extraction  

---

## 🧠 NLP Preprocessing Steps
1. **Text Cleaning:** Removing punctuation, special characters, numbers.  
2. **Lowercasing:** Converting all text to lowercase.  
3. **Stopword Removal:** Eliminating common non-informative words.  
4. **Tokenization:** Splitting sentences into tokens (words).  
5. **Lemmatization:** Reducing words to their base form.  
6. **Feature Extraction:** Using **TF-IDF Vectorization** to convert text to numerical form.

---

## 🤖 Models Implemented
| Model | Description |
|:------|:-------------|
| **Naive Bayes** | Baseline probabilistic classifier for text data. |
| **Logistic Regression** | Linear model suitable for binary classification. |
| **Support Vector Machine (SVM)** | High-dimensional classifier effective for textual data. |
| **Artificial Neural Network (ANN)** | Deep learning model built with Keras for advanced text feature learning. |

---

## 📊 Model Performance Comparison

### 🧩 Dataset 1: Job Train
| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|:------|:---------:|:----------:|:------:|:---------:|:----:|
| **ANN (Keras)** | **0.9731** | 0.8028 | 0.6263 | 0.7037 | 0.9359 |
| SVM (Linear) | 0.9692 | 0.7143 | 0.6593 | 0.6857 | 0.8705 |
| Logistic Regression | 0.9608 | 0.6000 | 0.6923 | 0.6428 | 0.9440 |
| Naive Bayes | 0.9044 | 0.3276 | 0.8351 | 0.4705 | 0.9329 |

---

### 🧩 Dataset 2: Fake Job Posting
| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|:------|:---------:|:----------:|:------:|:---------:|:----:|
| **ANN (Keras)** | **0.9840** | 0.9143 | 0.7399 | 0.8179 | 0.9877 |
| SVM (Linear) | 0.9829 | 0.8146 | 0.8382 | 0.8262 | 0.9751 |
| Logistic Regression | 0.9748 | 0.6948 | 0.8555 | 0.7668 | 0.9839 |
| Naive Bayes | 0.9155 | 0.3543 | 0.9133 | 0.5105 | 0.9747 |

---

## 📈 Key Insights
- The **ANN (Keras)** model achieved the **highest accuracy** across both datasets.  
- **SVM** also performed exceptionally well for text-based classification.  
- **Logistic Regression** gave stable results, while **Naive Bayes** showed high recall but low precision.  
- **TF-IDF feature extraction** played a crucial role in boosting performance.

---

## 💡 Conclusion
This project demonstrates how **NLP** and **machine learning** can be effectively combined to detect **fake job postings**.  
The **ANN model** outperformed other algorithms in terms of **accuracy and generalization**, making it the best choice for real-world deployment.

---

## 🚀 Future Enhancements
- Integrate **Word2Vec / BERT embeddings** for more contextual understanding.  
- Build a **Streamlit web app** for real-time job post verification.  
- Add **explainability (LIME, SHAP)** to make the model decisions interpretable.  
- Explore **deep transformers (BERT, RoBERTa)** for improved text analysis. 

---


