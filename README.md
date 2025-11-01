# 🧠 Fake Job Posting Detection Using Machine Learning & Deep Learning

## 🔍 Overview
This project focuses on detecting **fake job postings** using **Natural Language Processing (NLP)**, **Machine Learning (ML)**, and **Deep Learning (DL)** techniques.  
Fake job ads are widespread on the internet and can lead to fraud, scams, or personal data theft.  

The objective is to **automatically classify job postings** as **Real** or **Fake** using text-based analysis.  
Traditional ML models and Deep Learning models (based on **Artificial Neural Networks**) are combined to achieve **high accuracy and reliability**.  

A **Flask web application** is developed for real-time detection of fake job postings.  
🌐 **Live Demo:** [Fake Job Detection Web App (Vercel)](https://fake-job-detection-machine-learning-ten.vercel.app/)

---

## 📂 Dataset Sources

### 1️⃣ Fake Job Posting Dataset  
- **Source:** [Kaggle - Real or Fake Job Posting Prediction](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction)  
- **Size:** ~18,000 job postings  
- **Features:** Title, Company Profile, Description, Requirements, Benefits, etc.  
- **Target Variable:** `fraudulent` → (0 = Real, 1 = Fake)

### 2️⃣ Job Train Dataset  
- **Source:** Custom compiled dataset  
- **Size:** ~10,000 job postings  
- **Purpose:** To test model generalization and robustness on unseen data  

---

## 🧹 Data Preprocessing

### 🔸 Data Cleansing
- Removed duplicates and missing records.

### 🔸 Text Preprocessing
- Combined fields: *title, description, requirements, company profile*  
- Converted text to lowercase  
- Removed punctuation, symbols, and special characters  

### 🔸 Stopword Removal
- Applied using **NLTK Stopword Corpus**

### 🔸 Lemmatization & Tokenization
- Reduced words to their root forms for semantic clarity.  

### 🔸 Feature Extraction
- **TF-IDF Vectorization**  
  - `max_features = 5000`  
  - `ngram_range = (1, 2)`

### 🔸 Train-Test Split
- 80% training and 20% testing  
- Final dataset size: ~15,000–20,000 cleaned samples  

---

## 🤖 Models Implemented

| Model | Type | Description |
|--------|------|-------------|
| **Naive Bayes (Multinomial)** | ML | Lightweight probabilistic model ideal for text classification. |
| **Logistic Regression** | ML | Linear model with L2 regularization; interpretable and efficient. |
| **SVM (Linear Kernel)** | ML | Powerful classifier suitable for high-dimensional text data. |
| **ANN (Keras Sequential)** | DL | Deep neural network capturing complex text patterns. |

---

## 🧠 Artificial Neural Network (ANN) Architecture

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

### 📈 On Job Train Dataset

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|--------|-----------|------------|---------|-----------|------|
| **ANN (Keras)** | 97.31% | 0.7945 | 0.6374 | 0.7073 | 0.9359 |
| **SVM (Linear)** | 96.92% | 0.7143 | 0.6593 | 0.6857 | 0.8749 |
| **Logistic Regression** | 96.03% | 0.5962 | 0.6813 | 0.6359 | 0.9441 |
| **Naive Bayes** | 90.49% | 0.3290 | 0.8352 | 0.4720 | 0.9329 |

### 📉 On Fake Job Posting Dataset

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|--------|-----------|------------|---------|-----------|------|
| **ANN (Keras)** | 98.40% | 0.9143 | 0.7399 | 0.8179 | 0.9877 |
| **SVM (Linear)** | 98.29% | 0.8146 | 0.8382 | 0.8262 | 0.9751 |
| **Logistic Regression** | 97.48% | 0.6948 | 0.8555 | 0.7668 | 0.9839 |
| **Naive Bayes** | 91.55% | 0.3543 | 0.9133 | 0.5105 | 0.9747 |

---

## 📈 Visualizations

### 🔹 Confusion Matrices (ANN has the lowest false positives)
![Confusion Matrices](https://github.com/user-attachments/assets/82693aa7-f28e-407b-b0d0-cd301d068f9f)
![Confusion Matrices](https://github.com/user-attachments/assets/db89d28a-5673-4087-a9d7-f9a3ed2580eb)
![Confusion Matrices](https://github.com/user-attachments/assets/c64e35ea-de52-4aa2-b815-c1cc437250a6)
![Confusion Matrices](https://github.com/user-attachments/assets/91159901-5316-4c67-994f-08526ec9c526)

### 🔹 ROC Curves (ANN achieved AUC > 0.98)
![ROC Curve](https://github.com/user-attachments/assets/4afb3344-eaf1-411a-b137-93e3c229c519)

### 🔹 Performance Comparison
![Performance Comparison](https://github.com/user-attachments/assets/6544c892-b243-4394-8fa9-63df90c0259f)

---

## 🧪 Experiments & Results Summary

### ⚙️ Experimental Setup
- **Train-Test Split:** 80% training, 20% testing  
- **Cross-Validation:** 5-Fold CV  
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
- **Insight:** TF-IDF + ANN captures deep textual relationships effectively.  
- **Generalization:** Excellent performance across multiple datasets.  

---

## 🏁 Conclusion

This study demonstrates the potential of **AI and NLP** in combating **job posting fraud**.

### 🔑 Key Takeaways
- **ANN** achieves ~98% accuracy, outperforming traditional ML models.  
- **TF-IDF** is an effective representation method for text data.  
- The **Flask Web App** enables real-time fake job detection.  
- This methodology can be extended to:
  - Fake news detection  
  - Scam email filtering  
  - Online fraud analysis  

---

## ⚙️ Setup Instructions

### 🧩 Prerequisites
- **Python:** 3.8 or higher  
- **Environment:** Jupyter Notebook / VS Code  
- **Required Libraries:**  
  `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `keras`, `nltk`, `matplotlib`, `seaborn`

---

### 💾 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/fake-job-detection.git
cd fake-job-detection

# Install dependencies
pip install pandas numpy scikit-learn tensorflow keras nltk matplotlib seaborn
```

---

### 🚀 Execution Steps

#### 1️⃣ Data Preparation
```python
# Download datasets from Kaggle and place them in the project directory
# Files: fake_job_postings.csv and job_train.csv
```

#### 2️⃣ Run Model Training
```bash
# For Job Train dataset
jupyter notebook job_train.ipynb

# For Fake Job Posting dataset
jupyter notebook real_fake_job_posting.ipynb
```

#### 3️⃣ Launch Flask Web Application
```bash
python app.py
# Access the app locally at http://localhost:5000
```

#### 🌐 Live Deployed Version
👉 Visit here: [Fake Job Detection Web App (Vercel)](https://fake-job-detection-machine-learning-ten.vercel.app/)

#### 4️⃣ Model Files
Pre-trained models are stored as:
* `.pkl` → Machine Learning models  
* `.h5` → Artificial Neural Network (ANN) models  

---

## 📚 References

1. Kaggle Dataset: *Real or Fake Job Posting Prediction* - [Kaggle Link](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction)
2. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
3. Chollet, F. (2015). *Keras.* [https://keras.io](https://keras.io)
4. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python.* O'Reilly Media.
5. Zhang, Y., & Wallace, B. (2015). *A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification.* arXiv preprint arXiv:1510.03820.
6. Joachims, T. (1998). *Text categorization with support vector machines: Learning with many relevant features.* European Conference on Machine Learning, 137–142.
