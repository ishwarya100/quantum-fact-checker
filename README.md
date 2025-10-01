# Quantum-Assisted Misinformation Detection System

[![Open Live App](https://img.shields.io/badge/-Open%20App-orange?style=for-the-badge)](https://quantum-fact-checker.streamlit.app/)


This project is an AI-powered misinformation detection tool built with Streamlit.
It combines:

- Keyword & rule-based detection (quick heuristics)

- ML model (TF-IDF + Logistic Regression) trained on the Kaggle Fake News Dataset

- Transformers (BERT Zero-Shot Classification) for advanced text analysis (optional if installed)

- Quantum Simulation (Qiskit) to estimate misinformation spread factors

*Even if Transformers or Qiskit aren’t installed, the system still works with fallback logic.*

## Features

- Detects if a news post or social media text might contain misinformation.

- Uses Kaggle Fake News Dataset for training and higher accuracy.

- Optional BERT model for semantic misinformation classification.

- Simulates quantum-inspired spread factor for suspicious posts.

- Simple Streamlit Web App interface.

## Project Structure


```
Misinformation_Detection_System/
│
├── misinfo_detection.py   # Main Streamlit app
├── requirements.txt       # Required dependencies
├── README.md              # Project documentation
└── (optional) data/       # Local copy of dataset
```

## Installation
1. Clone the repo
   
   `git clone https://github.com/ishwarya100/quantum-fact-checker.git cd Misinformation_Detection_System`

2. Create a virtual environment
   
   `python -m venv venv`
   
   `source venv/bin/activate`   # Mac/Linux
   
   `venv\Scripts\activate`      # Windows

4. Install dependencies
   
   `pip install -r requirements.txt`


## Run 

Start the Streamlit app with:

`streamlit run misinfo_detection.py`


This will open the app in your browser.

## Usage

Enter a news headline or social media post in the text box.

Click Analyze.

The app shows:

- Misinformation Score (0–100)

- Quantum Spread Factor

- Prediction (Suspicious ✅ / Not Suspicious ⚠️)

- Dataset Model Accuracy (if dataset loaded successfully)

## Requirements

Python 3.8+

### Libraries:

streamlit

pandas

numpy

scikit-learn

matplotlib

networkx

transformers   # optional (for BERT)

qiskit         # optional (for quantum simulation)

------------------------------------------------------------------------------------------------------------------------------------------------------------------
### *Author*

#### *Developed by ISHWARYA.❤*
#### *AIML Enthusiast | Exploring Quantum + AI*
