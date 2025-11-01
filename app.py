from flask import Flask, request, jsonify, send_from_directory
import joblib
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

app = Flask(__name__)

# Load models and vectorizers for both datasets
# Dataset 1: job_train (using pkl files with '1' suffix)
model1 = joblib.load('logistic_model1.pkl')
tfidf1 = joblib.load('tfidf_vectorizer1.pkl')

# Dataset 2: fakejobposting (using pkl files without suffix)
model2 = joblib.load('logistic_model.pkl')
tfidf2 = joblib.load('tfidf_vectorizer.pkl')

# Download NLTK data if needed
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Clean text function (same as in notebook)
def clean_text(text):
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

@app.route('/styles.css')
def styles():
    return send_from_directory('frontend', 'styles.css')

@app.route('/predict', methods=['POST'])
def predict():
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

@app.route('/insights', methods=['GET'])
def insights():
    try:
        df = pd.read_csv('model_comparison.csv')
        data = df.to_dict(orient='records')
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)
