import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# NLP - Basic detection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Transformers (advanced NLP)
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ Transformers not available. Running in basic mode.")

# Quantum simulation
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    st.warning("⚠️ Qiskit not available. Using classical simulation only.")


# ================== LOAD DATASET (Kaggle Fake News) ==================
@st.cache_data
def load_and_train_model():
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/clmentbisaillon/fake_news_dataset/master/data.csv")
    except:
        return None, None   # gracefully return None if dataset not found

    # Preprocess dataset
    df = df.dropna(subset=['text', 'label'])
    X = df['text']
    y = df['label']  # 0 = true, 1 = fake

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline (TF-IDF + Logistic Regression)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipeline_model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', LogisticRegression(max_iter=300))
    ])

    pipeline_model.fit(X_train, y_train)
    accuracy = pipeline_model.score(X_test, y_test)

    return pipeline_model, accuracy


class MisinformationDetector:
    def __init__(self, use_bert=True):
        self.use_bert = use_bert and TRANSFORMERS_AVAILABLE
        self.pipeline_model, self.dataset_accuracy = load_and_train_model()

        self.suspicious_keywords = [
            'breaking', 'shocking', 'urgent', 'leaked', 'exposed',
            'they don\'t want you to know', 'hidden truth', 'conspiracy',
            'fake', 'hoax', 'scam', 'unbelievable', 'miracle cure',
            'doctors hate', 'secret', 'coverup', 'wake up'
        ]

        if self.use_bert:
            self.classifier = pipeline("zero-shot-classification",
                                       model="facebook/bart-large-mnli")

    def analyze_post(self, text):
        text = text.lower()
        score = 0

        # Keyword matching
        keyword_count = sum(1 for kw in self.suspicious_keywords if kw in text)
        score += keyword_count * 10

        # Sensationalism indicators
        if text.count('!') > 2:
            score += 15
        if text.isupper():
            score += 20
        if any(word in text for word in ['100%', 'guaranteed', 'proven']):
            score += 10

        word_count = len(text.split())
        if word_count < 10 or word_count > 200:
            score += 5

        # Advanced detection with BERT
        if self.use_bert:
            result = self.classifier(text, candidate_labels=["misinformation", "true information"])
            if "misinformation" in result['labels']:
                score += result['scores'][result['labels'].index("misinformation")] * 30

        # ML-based dataset prediction
        if self.pipeline_model:
            pred = self.pipeline_model.predict([text])[0]
            if pred == 1:  # fake news
                score += 25
            else:
                score -= 10

        score = min(max(score, 0), 100)

        # Quantum simulation (spread factor)
        spread_factor = self.simulate_quantum_spread()

        return score, score > 30, spread_factor

    def simulate_quantum_spread(self):
        if not QISKIT_AVAILABLE:
            return np.random.beta(2, 5)

        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(2, 'c')
        qc = QuantumCircuit(qr, cr)

        qc.h(qr[0])
        qc.h(qr[1])
        qc.cx(qr[0], qr[1])
        qc.h(qr[0])

        statevec = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
        probabilities = statevec.probabilities()
        return float(np.max(probabilities))


# ====================== STREAMLIT APP ======================
st.title("Quantum-Assisted Misinformation Detection System")

st.write("Enter a news post or social media text to check if it may contain misinformation.")

# User input
user_input = st.text_area("Enter text here:", height=150)

if st.button("Analyze"):
    if user_input.strip():
        detector = MisinformationDetector(use_bert=True)
        score, is_suspicious, spread_factor = detector.analyze_post(user_input)

        st.subheader("Analysis Result")
        st.write(f"**Misinformation Score:** {score:.2f}/100")
        st.write(f"**Quantum Spread Factor:** {spread_factor:.3f}")
        if detector.pipeline_model:
            st.write(f"**Dataset Model Accuracy:** {detector.dataset_accuracy*100:.2f}%")

        if is_suspicious:
            st.error("⚠️ This post looks suspicious and might contain misinformation!")
        else:
            st.success("✅ This post does not appear to contain misinformation.")
    else:
        st.warning("Please enter some text to analyze.")
