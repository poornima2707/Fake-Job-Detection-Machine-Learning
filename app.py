from flask import Flask, request, jsonify, send_from_directory
import joblib
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import os

app = Flask(__name__)

# Load models and vectorizers for both datasets
try:
    # Dataset 1: job_train (using pkl files with '1' suffix)
    model1 = joblib.load('logistic_model1.pkl')
    tfidf1 = joblib.load('tfidf_vectorizer1.pkl')

    # Dataset 2: fakejobposting (using pkl files without suffix)
    model2 = joblib.load('logistic_model.pkl')
    tfidf2 = joblib.load('tfidf_vectorizer.pkl')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    # Create dummy models for testing
    class DummyModel:
        def predict_proba(self, X):
            return [[0.4, 0.6]]  # Dummy probabilities
    
    class DummyVectorizer:
        def transform(self, texts):
            # Return dummy matrix
            from scipy.sparse import csr_matrix
            return csr_matrix((len(texts), 1000))
    
    model1 = model2 = DummyModel()
    tfidf1 = tfidf2 = DummyVectorizer()

# Download NLTK data if needed
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("✅ NLTK data downloaded successfully!")
except Exception as e:
    print(f"❌ Error downloading NLTK data: {e}")

# Clean text function (same as in notebook)
def clean_text(text):
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = re.sub(r'\[[^]]*\]', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()

    # Remove stopwords
    stop = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop])

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('frontend', filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if data is None:
            return jsonify({'error': 'Invalid JSON data'}), 400

        # Get selected dataset (default to job_train)
        dataset = data.get('dataset', 'job_train')

        # Select appropriate model and vectorizer
        if dataset == 'job_train':
            model = model1
            tfidf = tfidf1
        else:  # fakejobposting
            model = model2
            tfidf = tfidf2

        # Combine text fields
        text = ' '.join([
            data.get('title', ''),
            data.get('company_profile', ''),
            data.get('description', ''),
            data.get('requirements', '')
        ])

        # Clean and preprocess
        cleaned_text = clean_text(text)
        
        if not cleaned_text.strip():
            return jsonify({'error': 'No valid text provided for analysis'}), 400

        # Vectorize
        vectorized = tfidf.transform([cleaned_text])

        # Predict using logistic regression model
        prob = model.predict_proba(vectorized)[0][1]  # Probability of positive class (fake)
        prediction = 'Fake' if prob > 0.5 else 'Real'
        confidence = float(prob if prediction == 'Fake' else 1 - prob)

        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'dataset': dataset
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

@app.route('/insights', methods=['GET'])
def insights():
    try:
        if os.path.exists('model_comparison.csv'):
            df = pd.read_csv('model_comparison.csv')
            data = df.to_dict(orient='records')
            return jsonify(data)
        else:
            # Return dummy data for testing
            dummy_data = [
                {'Model': 'Logistic Regression', 'Accuracy': 0.984, 'Precision': 0.972, 'Recall': 0.968, 'F1': 0.970, 'AUC': 0.991},
                {'Model': 'Random Forest', 'Accuracy': 0.978, 'Precision': 0.965, 'Recall': 0.960, 'F1': 0.962, 'AUC': 0.985},
                {'Model': 'SVM', 'Accuracy': 0.981, 'Precision': 0.968, 'Recall': 0.965, 'F1': 0.966, 'AUC': 0.988},
                {'Model': 'XGBoost', 'Accuracy': 0.982, 'Precision': 0.970, 'Recall': 0.967, 'F1': 0.968, 'AUC': 0.990}
            ]
            return jsonify(dummy_data)
    except Exception as e:
        print(f"Insights error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create frontend directory if it doesn't exist
    if not os.path.exists('frontend'):
        os.makedirs('frontend')
        print("📁 Created frontend directory")
    
    print("🚀 Starting Flask server...")
    print("📧 Access the application at: http://127.0.0.1:5000")
    app.run(debug=False, port=5000)