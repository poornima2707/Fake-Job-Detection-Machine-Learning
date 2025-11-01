# 🧠 Fake Job Posting Detection Using Machine Learning & Deep Learning

## 🔍 Overview
This work focuses on detecting **fake job postings** using **Natural Language Processing (NLP)**, **Machine Learning (ML)**, and **Deep Learning (DL)** techniques.  
Fake job ads are widespread on the internet and can lead to fraud, scams, or personal data theft.  
The objective is to **automatically classify** job postings as **Real** or **Fake** using text-based analysis.  
Traditional ML models and Deep Learning models (based on **ANN**) are combined to achieve **high accuracy and reliability**.  
A **Flask web application** is also developed for real-time detection of fake job postings.

---

## 📂 Dataset Sources

### 1. Fake Job Posting Dataset  
- **Source:** Kaggle - *Real or Fake Job Posting Prediction*  
- **Size:** ~18,000 job postings  
- **Features:** Title, Company Profile, Description, Requirements, Benefits, etc.  
- **Target Variable:** `fraudulent` → (0 = Real, 1 = Fake)

### 2. Job Train Dataset  
- **Source:** Custom compiled dataset  
- **Size:** ~10,000 job postings  
- **Goal:** Test model generalization and robustness on unseen data  

---

## 🧹 Data Preprocessing

### Data Cleansing
- Removed duplicate and missing records  

### Text Preprocessing
- Combined key fields: *title, description, requirements, company profile*  
- Converted text to lowercase  
- Removed punctuation, symbols, and special characters  

### Stopword Removal
- Applied using **NLTK Stopword Corpus**

### Lemmatization and Tokenization
- Reduced words to their root form  

### Feature Extraction
- **TF-IDF Vectorization**  
  - `max_features = 5000`  
  - `ngram_range = (1, 2)`

### Train-Test Split
- 80% training and 20% testing  
- Final dataset size: ~15,000–20,000 cleaned samples  

---

## 🤖 Models Implemented

| Model | Type | Description |
|--------|------|-------------|
| **Naive Bayes (Multinomial)** | ML | Lightweight and fast probabilistic classifier, ideal for text classification. |
| **Logistic Regression** | ML | Linear model with L2 regularization; interpretable coefficients for insights. |
| **SVM (Linear Kernel)** | ML | Powerful classifier suitable for high-dimensional data. |
| **ANN (Keras Sequential)** | DL | Feedforward neural network with hidden layers, ReLU activation, and dropout. |

---

## 🧠 Artificial Neural Network Architecture

- **Input Layer:** TF-IDF feature vector  
- **Hidden Layers:**  
  - Dense (128 neurons, ReLU activation)  
  - Dropout (0.3)  
- **Output Layer:** Dense (1 neuron, Sigmoid activation)  

**Optimizer:** Adam (learning rate = 0.001)  
**Loss Function:** Binary Cross-Entropy  
**Metrics:** Accuracy, AUC  

---

## ⚙️ Training Setup

- **Frameworks Used:** scikit-learn, TensorFlow, Keras, NLTK, Matplotlib, Seaborn  
- **Validation Strategy:** 5-Fold Cross Validation  
- **Batch Size:** 32  
- **Epochs:** 50  
- **Learning Rate:** 0.001  

---

## 📊 Performance Comparison

### On Job Train Dataset

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|--------|-----------|------------|---------|-----------|------|
| **ANN (Keras)** | 97.31% | 0.7945 | 0.6374 | 0.7073 | 0.9359 |
| **SVM (Linear)** | 96.92% | 0.7143 | 0.6593 | 0.6857 | 0.8749 |
| **Logistic Regression** | 96.03% | 0.5962 | 0.6813 | 0.6359 | 0.9441 |
| **Naive Bayes** | 90.49% | 0.3290 | 0.8352 | 0.4720 | 0.9329 |

### On Fake Job Posting Dataset

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|--------|-----------|------------|---------|-----------|------|
| **ANN (Keras)** | 98.40% | 0.9143 | 0.7399 | 0.8179 | 0.9877 |
| **SVM (Linear)** | 98.29% | 0.8146 | 0.8382 | 0.8262 | 0.9751 |
| **Logistic Regression** | 97.48% | 0.6948 | 0.8555 | 0.7668 | 0.9839 |
| **Naive Bayes** | 91.55% | 0.3543 | 0.9133 | 0.5105 | 0.9747 |

---

## 📈 Visualizations

- **Confusion Matrices:** ANN has the lowest false positives  
![Confusion Matrix 1](https://github.com/user-attachments/assets/5861aed7-9e17-4021-b1f9-a0561f6a402e)  
![Confusion Matrix 2](https://github.com/user-attachments/assets/8d12e3e3-167a-4c00-b936-e892a2b32907)  
![Confusion Matrix 3](https://github.com/user-attachments/assets/017c1549-9d0a-4231-b083-7e42afbaed3d)  
![Confusion Matrix 4](https://github.com/user-attachments/assets/c88f3e0b-3288-423a-bdd2-9708229d8024)

- **ROC Curves:** ANN achieved AUC > 0.98 on both datasets  
![ROC Curves](https://github.com/user-attachments/assets/c2072ddc-0ad7-4ba3-a5e4-0a197e6f6ee3)

- **Performance Comparison:** ANN outperformed other models across all metrics  
![Performance Comparison](https://github.com/user-attachments/assets/3c46262c-a807-4b68-9673-b1042b3a99d8)

---

## 📊 Experiments & Results Summary

### Experimental Setup
- **Train-Test Split:** 80% training, 20% testing  
- **Cross-Validation:** 5-fold CV for hyperparameter tuning  
- **Hyperparameters:**  
  - **TF-IDF:** `max_features=5000`, `ngram_range=(1,2)`  
  - **SVM:** `C=1.0`, `kernel='linear'`  
  - **Logistic Regression:** `C=1.0`, `penalty='l2'`  
  - **ANN:** `epochs=50`, `batch_size=32`, `learning_rate=0.001`  

---

## ✅ Results & Insights

- **Best Model:** ANN (Keras)  
- **Highest Accuracy:** 98.4%  
- **Best AUC:** 0.9877  
- **Key Insight:** TF-IDF + ANN captures deep textual relationships effectively  
- **Generalization:** Excellent performance across multiple datasets  

---

## ✅ Conclusion

This work demonstrates the potential of **AI and NLP** in combating **job posting fraud**.  

### 🔑 Key Takeaways
- **ANN** achieves ~98% accuracy, outperforming traditional ML models  
- **TF-IDF** is an effective representation method for textual data  
- The **Flask Web App** allows real-time fake job detection  
- This methodology can be extended to:
  - Fake news detection  
  - Scam email filtering  
  - Online fraud analysis  

---

## ⚙️ Setup Instructions

### Prerequisites
- **Python:** 3.8 or higher  
- **Environment:** Jupyter Notebook / VS Code  
- **Required Libraries:** pandas, numpy, scikit-learn, tensorflow, keras, nltk, matplotlib, seaborn  

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/fake-job-detection.git
cd fake-job-detection

# Install dependencies
pip install pandas numpy scikit-learn tensorflow keras nltk matplotlib seaborn
```

---

## ⚙️ Execution Steps

### **1. Data Preparation**

```python
# Download datasets from Kaggle and place them in the project directory
# Files: fake_job_postings.csv and job_train.csv
```

---

### **2. Run Model Training**

```bash
# For Job Train dataset
jupyter notebook job_train.ipynb

# For Fake Job Posting dataset
jupyter notebook real_fake_job_posting.ipynb
```

---

### **3️⃣ Launch Flask Web Application**

```bash
python app.py
# Access the app at http://localhost:5000
```

---

### **4️⃣ Model Files**

Pre-trained models are stored as:

- `.pkl` → Machine Learning models  
- `.h5` → Artificial Neural Network (ANN) models  

---

## 📚 References

1. **Kaggle Dataset** – *Real or Fake Job Posting Prediction*  
2. **Scikit-learn Documentation**  
3. **TensorFlow Keras API**  
4. **Bird, Klein & Loper (2009)** – *Natural Language Processing with Python*  
5. **Zhang & Wallace (2015)** – *CNN Sensitivity Analysis for Sentence Classification*
