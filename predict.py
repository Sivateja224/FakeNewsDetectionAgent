# predict.py
# A standalone CLI helper script to predict if a news article is Fake or Real.
# This script loads the trained model and vectorizer, preprocesses the input text,
# and outputs the prediction, confidence score, reason, and recommendation.

import os
import re
import math
import joblib
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Lemmatizer and Stopwords list
# We download them if not already present (failsafe for new systems)
nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

try:
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
    nltk.download('omw-1.4', download_dir=nltk_data_dir, quiet=True)
except Exception:
    pass

lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set()

def preprocess_text(text):
    """
    Cleans and preprocesses the text. Must match the logic in train_model.py exactly.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

def analyze_suspicious_patterns(text):
    """
    Analyzes the text for typical patterns found in fake news:
    - Sensational words (clickbait)
    - Excessive exclamation marks or question marks
    - Words in all caps (SHOUTING)
    """
    patterns = []
    
    # Check for clickbait/sensational keywords
    sensational_keywords = [
        "shocking", "unbelievable", "secret", "exposed", "must see",
        "conspiracy", "miracle", "won't believe", "classified", "truth about",
        "scandal", "proof", "alert", "breaking news", "conspirators"
    ]
    text_lower = text.lower()
    found_keywords = [kw for kw in sensational_keywords if kw in text_lower]
    if found_keywords:
        patterns.append(f"Sensational keywords detected: {', '.join(found_keywords)}")
    
    # Check for excessive exclamation marks
    excl_count = text.count("!")
    if excl_count > 3:
        patterns.append(f"Excessive use of exclamation marks ({excl_count} found)")
        
    # Check for words in ALL CAPS (shouting) that are not short acronyms (length > 4)
    shouting_words = [w for w in text.split() if w.isupper() and len(w) > 4]
    if shouting_words:
        patterns.append(f"Words in ALL CAPS (shouting style): {', '.join(shouting_words[:5])}")
        
    return patterns

def generate_ai_explanation(prediction, confidence, patterns):
    """
    Generates an explanation of the prediction in simple language,
    acting like an AI assistant.
    """
    if prediction == 0:  # Fake News
        explanation = (
            f"The article contains language patterns and vocabulary that closely align with previously "
            f"classified fake news in my training database. Specifically, it relies on styling or key phrases "
            f"that are mathematically associated with low-credibility writing."
        )
        if patterns:
            explanation += f" Additionally, the following warning signs were identified: {'; '.join(patterns)}."
    else:  # Real News
        explanation = (
            f"The article's text, syntax, and word choices align with typical patterns found in verified, "
            f"reliable journalism in my training database. It lacks the exaggerated styling, sensational "
            f"markers, or lexical patterns common in misinformation."
        )
        if patterns:
            explanation += f" However, note these subtle styling markers: {'; '.join(patterns)}."
            
    return explanation

def generate_recommendation(prediction):
    """
    Generates action recommendations for the user.
    """
    if prediction == 0:  # Fake News
        return (
            "This article exhibits several characteristics commonly associated with misleading or fabricated news. "
            "Before sharing or acting on this information, verify it using trusted sources such as Reuters, AP News, "
            "BBC, or official government websites."
        )
    else:  # Real News
        return (
            "RECOMMENDATION: While this article appears credible, it is always a good practice to verify key facts "
            "with primary sources or official government/scientific bulletins."
        )

def run_prediction(text, model_path="model.pkl", vectorizer_path="vectorizer.pkl"):
    """
    Loads model/vectorizer, runs preprocessing and returns the full result dictionary.
    """
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return {
            "error": "Trained model and/or vectorizer not found. Please run 'train_model.py' first to generate them."
        }
        
    # Load model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Vectorize
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = int(model.predict(vectorized_text)[0])
    
    # Calculate Confidence Score using Sigmoid function on the decision_function score
    decision_score = float(model.decision_function(vectorized_text)[0])
    # The sigmoid output maps (-inf, inf) -> (0, 1).
    # Since confidence is how sure the model is about its output class,
    # we take the absolute value of decision score so the sigmoid is always >= 0.5 (representing 50% to 100% confidence).
    confidence = 1 / (1 + math.exp(-abs(decision_score)))
    
    # Analyze warning patterns
    patterns = analyze_suspicious_patterns(text)
    
    # Generate explanations
    explanation = generate_ai_explanation(prediction, confidence, patterns)
    recommendation = generate_recommendation(prediction)
    
    return {
        "prediction_code": prediction,
        "prediction": "Real News" if prediction == 1 else "Fake News",
        "confidence_percentage": round(confidence * 100, 2),
        "explanation": explanation,
        "recommendation": recommendation,
        "suspicious_patterns": patterns
    }

def main():
    # CLI user prompt if no arguments are provided
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        print("=== Fake News Detection Agent (CLI) ===")
        print("Enter or paste your news article below. Press Ctrl+Z (Windows) or Ctrl+D (Unix) then Enter to submit:\n")
        text = sys.stdin.read().strip()
        
    if not text:
        print("Error: No input text provided.")
        sys.exit(1)
        
    print("\n--- Analysing article... ---\n")
    result = run_prediction(text)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
        
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence_percentage']}%")
    print(f"Reason:     {result['explanation']}")
    print(f"Recommendation: {result['recommendation']}")
    
    if result['suspicious_patterns']:
        print("\nFlagged Style Indicators:")
        for pattern in result['suspicious_patterns']:
            print(f" - {pattern}")

if __name__ == "__main__":
    main()
