# app.py
# The main Flask web application for the Fake News Detection Agent.
# It serves the frontend index page and handles incoming API requests for news classification.

import os
from flask import Flask, request, jsonify, render_template

# Import our custom prediction helper
# This ensures that both the CLI and web app share identical preprocessing and model logic.
from predict import run_prediction

# Initialise the Flask Application
app = Flask(__name__)

# Ensure paths align with the folder structure
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

@app.route("/")
def home():
    """
    Renders the main dashboard page.
    """
    # Check if the models exist and warn in console if they aren't trained yet
    model_exists = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
    return render_template("index.html", model_trained=model_exists)

@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint that accepts news text and returns the classification results.
    Expects JSON payload: { "text": "Your article content..." }
    """
    try:
        # Parse JSON request data
        data = request.get_json()
        
        # Check if the request contains valid data
        if not data or "text" not in data:
            return jsonify({
                "status": "error",
                "message": "Invalid request. Please provide JSON data containing a 'text' key."
            }), 400
            
        text = data["text"].strip()
        
        # Validate input text length
        if not text:
            return jsonify({
                "status": "error",
                "message": "Please enter a news article to analyze."
            }), 400
            
        # Run prediction pipeline
        result = run_prediction(text, MODEL_PATH, VECTORIZER_PATH)
        
        # Handle model load errors or missing files
        if "error" in result:
            return jsonify({
                "status": "error",
                "message": result["error"]
            }), 500
            
        # Return results to client
        return jsonify({
            "status": "success",
            "prediction": result["prediction"],
            "prediction_code": result["prediction_code"],
            "confidence": result["confidence_percentage"],
            "explanation": result["explanation"],
            "recommendation": result["recommendation"],
            "suspicious_patterns": result["suspicious_patterns"]
        })
        
    except Exception as e:
        # Gracefully handle unexpected exceptions
        print(f"Exception during prediction API call: {e}")
        return jsonify({
            "status": "error",
            "message": "An unexpected server error occurred. Please try again."
        }), 500

if __name__ == "__main__":
    # Start the Flask development server
    # Running on port 5000 by default, debug mode activated for easy feedback
    print("Starting Fake News Detection Agent Web Server...")
    import threading
    if threading.current_thread() is threading.main_thread():
        app.run(debug=True, host="127.0.0.1", port=5000)
    else:
        print("Flask web server launch skipped because it is not running in the main thread (likely run via Streamlit).")

