# streamlit_app.py
# A beautiful, interactive Streamlit application for the Fake News Detection Agent.
# Suitable for local execution and seamless deployment to Streamlit Community Cloud.

import os
import re
import math
import joblib
import pandas as pd
import numpy as np
import nltk
import streamlit as st
from datetime import datetime

# Import preprocessing and prediction logic from predict.py
from predict import run_prediction, preprocess_text, analyze_suspicious_patterns

# Configure NLTK data path for deployment environments (e.g. Streamlit Cloud)
nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Page Configuration
st.set_page_config(
    page_title="Fake News Detection Agent - AI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism & Radial Gradient Background)
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global font and background override */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #03001e !important;
        background-image: radial-gradient(circle at 10% 20%, rgb(4, 12, 36) 0%, rgb(16, 37, 76) 45%, rgb(4, 12, 36) 90%) !important;
        color: #f8f9fa !important;
    }

    /* Streamlit block containers */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Titles and headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Custom Glassmorphism Card Wrapper */
    .glass-card {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px) saturate(120%);
        -webkit-backdrop-filter: blur(12px) saturate(120%);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 1rem;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    /* Info card sub-boxes */
    .bg-white-10 {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 0.75rem;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .bg-danger-10 {
        background: rgba(239, 68, 68, 0.08) !important;
        border: 1px solid rgba(239, 68, 68, 0.2) !important;
        border-radius: 0.75rem;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .bg-warning-10 {
        background: rgba(245, 158, 11, 0.08) !important;
        border: 1px solid rgba(245, 158, 11, 0.2) !important;
        border-radius: 0.75rem;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    /* Style the Streamlit text area */
    .stTextArea textarea {
        background: rgba(0, 0, 0, 0.25) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        border-radius: 0.75rem !important;
        padding: 1.2rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.3) !important;
    }

    /* Styled buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3) !important;
        color: #ffffff !important;
    }

    div.stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Clean clear buttons */
    .clear-btn > div.stButton > button {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    .clear-btn > div.stButton > button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
        transform: translateY(-2px) !important;
        box-shadow: none !important;
    }

    /* Sidebar styling override */
    [data-testid="stSidebar"] {
        background-color: rgba(4, 12, 36, 0.85) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* Result Badges */
    .result-badge-container {
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .badge-real {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important;
        padding: 0.6rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.25rem;
        display: inline-block;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3);
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .badge-fake {
        background: linear-gradient(135deg, #ef4444, #dc2626) !important;
        color: white !important;
        padding: 0.6rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.25rem;
        display: inline-block;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.3);
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* Progress bar elements styling */
    .confidence-label {
        font-weight: 600;
        color: #e2e8f0;
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }

    .custom-progress {
        background-color: rgba(0, 0, 0, 0.4);
        border-radius: 10px;
        height: 14px;
        overflow: hidden;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .custom-progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .custom-progress-bar.real {
        background: linear-gradient(90deg, #10b981, #34d399);
    }
    
    .custom-progress-bar.fake {
        background: linear-gradient(90deg, #ef4444, #f87171);
    }
    
    /* History lists custom scrollbar */
    .history-container {
        max-height: 250px;
        overflow-y: auto;
        padding-right: 5px;
    }
    .history-container::-webkit-scrollbar {
        width: 6px;
    }
    .history-container::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 4px;
    }
    .history-container::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 4px;
    }
    
    /* Sidebar info widgets */
    .sidebar-widget {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        font-size: 0.85rem;
    }
    
    /* Interactive session list item button styling */
    .session-item-btn {
        display: block;
        width: 100%;
        text-align: left;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #cbd5e1;
        padding: 0.5rem 0.75rem;
        border-radius: 0.375rem;
        margin-bottom: 0.5rem;
        font-size: 0.8rem;
        cursor: pointer;
        transition: all 0.2s ease;
        text-overflow: ellipsis;
        white-space: nowrap;
        overflow: hidden;
    }
    .session-item-btn:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.15);
        color: #ffffff;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# File Paths
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
DATASET_DIR = "dataset"
FAKE_PATH = os.path.join(DATASET_DIR, "Fake.csv")
TRUE_PATH = os.path.join(DATASET_DIR, "True.csv")

# Helper function to check if models are trained
def check_models_trained():
    return os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)

# Helper function to run the model training
def train_model_pipeline():
    try:
        # Download NLTK resources to local folder
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
        nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
        nltk.download('omw-1.4', download_dir=nltk_data_dir, quiet=True)
        
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        # Load datasets
        if not os.path.exists(FAKE_PATH) or not os.path.exists(TRUE_PATH):
            return {"status": "error", "message": "Dataset files not found in dataset/. Please place Fake.csv and True.csv there."}
            
        progress_text = st.empty()
        progress_bar = st.progress(0.0)
        
        progress_text.text("Step 1/5: Loading dataset CSVs...")
        progress_bar.progress(0.1)
        fake_df = pd.read_csv(FAKE_PATH)
        true_df = pd.read_csv(TRUE_PATH)
        
        fake_df['label'] = 0
        true_df['label'] = 1
        
        df = pd.concat([fake_df, true_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.dropna(subset=['title', 'text'], inplace=True)
        df['full_text'] = df['title'] + " " + df['text']
        
        # Preprocessing text (limit to 15,000 records for quick compilation if resource constrained, 
        # or use full if requested. We do full as train_model.py does but with batches)
        progress_text.text("Step 2/5: Cleaning and lemmatizing text (this takes about 1 min)...")
        progress_bar.progress(0.3)
        
        total_records = len(df)
        processed_texts = []
        chunk_size = max(total_records // 10, 1)
        for idx, text in enumerate(df['full_text']):
            processed_texts.append(preprocess_text(text))
            if (idx + 1) % chunk_size == 0 or idx == total_records - 1:
                percentage = int(((idx + 1) / total_records) * 100)
                progress_text.text(f"Step 2/5: Preprocessing... {percentage}% completed")
                progress_bar.progress(0.3 + (percentage / 100.0) * 0.4) # scale between 0.3 and 0.7
                
        df['cleaned_text'] = processed_texts
        
        progress_text.text("Step 3/5: Running TF-IDF Vectorization...")
        progress_bar.progress(0.75)
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df['cleaned_text'])
        y = df['label'].values
        
        progress_text.text("Step 4/5: Splitting datasets & training classifier...")
        progress_bar.progress(0.85)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        pac = SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1', eta0=1.0, max_iter=50, random_state=42)
        pac.fit(X_train, y_train)
        
        # Evaluate
        progress_text.text("Step 5/5: Evaluating model metrics and saving to disk...")
        progress_bar.progress(0.95)
        y_pred = pac.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Save to disk
        joblib.dump(pac, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        
        progress_bar.progress(1.0)
        progress_text.empty()
        
        return {
            "status": "success",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    except Exception as e:
        return {"status": "error", "message": f"Training failed: {str(e)}"}

# Presets Definition
PRESETS = {
    "Real News Example": (
        "The Federal Reserve kept its benchmark interest rate unchanged on Wednesday, pointing to solid economic growth "
        "and a strong job market, while indicating that inflation remains elevated. In its post-meeting statement, the "
        "central bank noted that although inflation has eased over the past year, it still remains above the Fed's 2% target. "
        "Fed Chair Jerome Powell stated during a press conference that the committee needs greater confidence that inflation "
        "is moving sustainably toward their target before reducing the policy rate. Economists expect the Fed to maintain "
        "this cautious stance for the coming months."
    ),
    "Fake News Example": (
        "SHOCKING CONSPIRACY EXPOSED: Secret government documents leaked online prove a shocking miracle cure for all diseases "
        "is being actively suppressed by high-ranking conspirators! You must see this unbelievable proof before it gets taken "
        "down! For years, major pharmaceutical corporations have collaborated with corrupt politicians to hide this truth "
        "from the public to keep profits soaring. Alert your family and friends immediately! Share this breaking news everywhere "
        "before it is classified forever!"
    )
}

# Session State Initialization
if "history" not in st.session_state:
    st.session_state["history"] = []
if "current_text" not in st.session_state:
    st.session_state["current_text"] = ""
if "active_preset" not in st.session_state:
    st.session_state["active_preset"] = "Custom Text"
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None

# Header layout
col1, col2 = st.columns([1, 10])
with col1:
    st.markdown("<h1 style='text-align: center; margin-top: 0.5rem;'>🤖</h1>", unsafe_allow_html=True)
with col2:
    st.title("Fake News Detection Agent")
    st.markdown("<p style='font-size: 1.15rem; color: rgba(255, 255, 255, 0.7); margin-top: -0.75rem;'>An Agentic AI assistant that detects misinformation using Machine Learning.</p>", unsafe_allow_html=True)

st.write("---")

# Main Section Splits (Left Sidebar, Center/Right tabs)
# Sidebar controls
with st.sidebar:
    st.markdown("### 🛠️ Model Dashboard")
    
    # Model Status Card
    model_trained = check_models_trained()
    if model_trained:
        st.markdown(
            '<div class="sidebar-widget" style="border-left: 4px solid #10b981;">'
            '🟢 <strong>Model Status:</strong> Trained & Loaded<br>'
            '<small style="color: rgba(255, 255, 255, 0.5);">Classifier: Passive Aggressive</small>'
            '</div>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="sidebar-widget" style="border-left: 4px solid #f59e0b; background: rgba(245,158,11,0.05);">'
            '⚠️ <strong>Model Status:</strong> Missing trained files!<br>'
            '<small>Please click the button below to train a new model from the dataset.</small>'
            '</div>', 
            unsafe_allow_html=True
        )
        
    # Retrain button
    if st.button("🔄 Train Model" if not model_trained else "🔄 Retrain Classifier"):
        with st.status("Training Machine Learning Model...", expanded=True) as status:
            result = train_model_pipeline()
            if result["status"] == "success":
                status.update(label="Model Trained Successfully!", state="complete", expanded=False)
                st.success(f"Model saved successfully! Accuracy: {result['accuracy']*100:.2f}%")
                st.rerun()
            else:
                status.update(label="Training Failed!", state="error")
                st.error(result["message"])

    st.markdown("---")
    st.markdown("### 🕒 Session History")
    
    if st.session_state["history"]:
        # Add clear button
        if st.button("🗑️ Clear History", key="clear_history_btn"):
            st.session_state["history"] = []
            st.rerun()
            
        st.markdown('<div class="history-container">', unsafe_allow_html=True)
        for idx, item in enumerate(st.session_state["history"]):
            label = "🟢 Real" if item["result"]["prediction_code"] == 1 else "🔴 Fake"
            timestamp = item["time"]
            preview = item["text"][:30] + "..."
            
            # Use unique key for buttons
            if st.button(f"{label} | {preview}", key=f"hist_{idx}", help=item["text"]):
                st.session_state["current_text"] = item["text"]
                st.session_state["analysis_result"] = item["result"]
                st.session_state["active_preset"] = "Custom Text"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: rgba(255, 255, 255, 0.4); font-size: 0.85rem; text-align: center; padding: 1rem 0;'>No articles analyzed in this session yet.</p>", unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### 💡 Help & Reference")
    st.markdown(
        "<div class='sidebar-widget' style='background: rgba(255,255,255,0.01);'>"
        "<strong>How it works:</strong> The agent cleans your text (removing stopwords/lemmatizing) "
        "and runs it through a 5,000-feature TF-IDF Vectorizer combined with a Passive Aggressive Classifier. "
        "Warning style rules check for shouting text, clickbait titles, and sensationalized punctuation."
        "</div>",
        unsafe_allow_html=True
    )

# Setup Tabs
tab_detector, tab_metrics, tab_about = st.tabs(["🔍 Detector Dashboard", "📊 Model Insights", "ℹ️ About the Agent"])

# Tab 1: Detector Dashboard
with tab_detector:
    # Model Warning at top of tab if not trained
    if not model_trained:
        st.warning("⚠️ Machine Learning model files (model.pkl, vectorizer.pkl) are not detected in the project directory. Please click 'Train Model' in the sidebar to build the model files.")

    # Two columns: Input (Left) and Results (Right)
    left_col, right_col = st.columns([6, 5])
    
    # Left Column: Input and Control
    with left_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📰 Analyze News Article")
        st.write("Paste the complete text or headline of a news article below to let the AI agent inspect its structure, tone, and credibility.")
        
        # Presets selector
        selected_preset = st.selectbox(
            "Select an Example Preset:",
            ["Custom Text", "Real News Example", "Fake News Example"],
            index=["Custom Text", "Real News Example", "Fake News Example"].index(st.session_state["active_preset"])
        )
        
        # Set text based on selection
        if selected_preset != st.session_state["active_preset"]:
            st.session_state["active_preset"] = selected_preset
            if selected_preset == "Custom Text":
                st.session_state["current_text"] = ""
            else:
                st.session_state["current_text"] = PRESETS[selected_preset]
            # Reset prediction state on preset change
            st.session_state["analysis_result"] = None
            st.rerun()

        # Text input box
        input_text = st.text_area(
            "Article Body / Headline:",
            value=st.session_state["current_text"],
            placeholder="Paste article content here (at least 10 words for best accuracy)...",
            height=300
        )
        
        # Keep input text in state
        st.session_state["current_text"] = input_text

        # Character/Word count utility
        words_count = len(input_text.split()) if input_text.strip() else 0
        char_count = len(input_text)
        st.markdown(f"<p style='text-align: right; color: rgba(255,255,255,0.4); font-size: 0.85rem; margin-top: -0.5rem;'>Words: {words_count} | Characters: {char_count}</p>", unsafe_allow_html=True)
        
        # Action Buttons row
        btn_col1, btn_col2 = st.columns([4, 1])
        with btn_col1:
            analyze_clicked = st.button("🚀 Analyze & Detect Credibility")
        with btn_col2:
            st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
            clear_clicked = st.button("🗑️ Clear")
            st.markdown('</div>', unsafe_allow_html=True)
            
        if clear_clicked:
            st.session_state["current_text"] = ""
            st.session_state["analysis_result"] = None
            st.session_state["active_preset"] = "Custom Text"
            st.rerun()
            
        if analyze_clicked:
            if not model_trained:
                st.error("Cannot predict: The model must be trained first. Click 'Train Model' in the sidebar.")
            elif not input_text.strip():
                st.warning("Please enter a news article to analyze.")
            elif words_count < 5:
                st.warning("Please paste a longer text (at least 5 words) for an accurate analysis.")
            else:
                # Run prediction
                with st.spinner("Analyzing linguistic patterns..."):
                    result = run_prediction(input_text, MODEL_PATH, VECTORIZER_PATH)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.session_state["analysis_result"] = result
                        # Add to history
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        st.session_state["history"].insert(0, {
                            "text": input_text,
                            "result": result,
                            "time": timestamp
                        })
                        st.rerun()
                        
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Right Column: Results Dashboard
    with right_col:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("📊 Analysis Output")
        
        res = st.session_state["analysis_result"]
        
        if res is None:
            # Placeholder Empty State
            st.markdown(
                '<div style="text-align: center; padding: 5rem 0; color: rgba(255, 255, 255, 0.4);">'
                '<h1 style="font-size: 5rem; margin-bottom: 1rem; opacity: 0.5;">📋</h1>'
                '<h4>Awaiting Input</h4>'
                '<p>Enter a news article on the left and click "Analyze & Detect" to see the AI analysis output.</p>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            # Result Badge
            pred_class = res["prediction"]
            pred_code = res["prediction_code"]
            conf = res["confidence_percentage"]
            
            badge_class = "badge-real" if pred_code == 1 else "badge-fake"
            
            st.markdown(
                f'<div class="result-badge-container">'
                f'<span class="{badge_class}">{pred_class}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Confidence Progress Bar
            progress_bar_color_class = "real" if pred_code == 1 else "fake"
            st.markdown(
                f'<div class="confidence-label">'
                f'<span>Agent Confidence Score:</span>'
                f'<span style="color: #60a5fa; font-weight: 700;">{conf}%</span>'
                f'</div>'
                f'<div class="custom-progress">'
                f'<div class="custom-progress-bar {progress_bar_color_class}" style="width: {conf}%;"></div>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # AI Explanation
            st.markdown(
                f'<div class="bg-white-10">'
                f'<h6 style="color: #60a5fa; margin-bottom: 0.5rem; display: flex; align-items: center;">'
                f'<span style="margin-right: 0.5rem;">💬</span>AI Reasoning & Explanation'
                f'</h6>'
                f'<p style="margin: 0; font-size: 0.9rem; line-height: 1.5; color: #e2e8f0;">{res["explanation"]}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Suspicious Patterns Callout (if any exist)
            patterns = res["suspicious_patterns"]
            if patterns:
                patterns_list = "".join([f"<li style='margin-bottom: 0.25rem;'>🚨 {p}</li>" for p in patterns])
                st.markdown(
                    f'<div class="bg-danger-10">'
                    f'<h6 style="color: #ef4444; margin-bottom: 0.5rem; display: flex; align-items: center;">'
                    f'<span style="margin-right: 0.5rem;">⚠️</span>Flagged Style Indicators ({len(patterns)})'
                    f'</h6>'
                    f'<ul style="margin: 0; padding-left: 1.2rem; font-size: 0.88rem; color: #fca5a5;">'
                    f'{patterns_list}'
                    f'</ul>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
            # Recommendations Card
            rec_color = "#34d399" if pred_code == 1 else "#fbbf24"
            rec_icon = "🛡️" if pred_code == 1 else "⚠️"
            st.markdown(
                f'<div class="bg-warning-10">'
                f'<h6 style="color: {rec_color}; margin-bottom: 0.5rem; display: flex; align-items: center;">'
                f'<span style="margin-right: 0.5rem;">{rec_icon}</span>Recommendation'
                f'</h6>'
                f'<p style="margin: 0; font-size: 0.9rem; line-height: 1.5; color: #fde68a;">{res["recommendation"]}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Model Performance & Insights
with tab_metrics:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Classifier Insights & Architecture")
    st.write("Below are the configuration details, training architecture, and baseline metrics of the Active Passive Classifier (PAC) model deployed inside the agent.")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric(label="Model Accuracy", value="99.28%")
    with col_stat2:
        st.metric(label="Precision Score", value="99.16%")
    with col_stat3:
        st.metric(label="Recall Score", value="99.41%")
    with col_stat4:
        st.metric(label="F1 Score", value="99.28%")
        
    st.markdown("---")
    
    st.markdown("#### How the Pipeline Works")
    
    col_pipe1, col_pipe2, col_pipe3 = st.columns(3)
    with col_pipe1:
        st.markdown(
            "##### 1. Text Cleanup"
            "\n- Converts all input characters to lowercase."
            "\n- Strips punctuation and digits."
            "\n- Filters out standard English stopwords."
            "\n- Lemmatizes tokens to their root dictionary form."
        )
    with col_pipe2:
        st.markdown(
            "##### 2. TF-IDF Vectorization"
            "\n- Vocabulary restricted to the top 5,000 features."
            "\n- Captures single words (unigrams) and double words (bigrams)."
            "\n- Weighs terms by Term Frequency-Inverse Document Frequency to find rare, identifying tokens."
        )
    with col_pipe3:
        st.markdown(
            "##### 3. Classifier Weights"
            "\n- Uses a Passive Aggressive Classifier (PAC)."
            "\n- If a classification is correct, the model remains passive (no changes to weights)."
            "\n- If incorrect, it aggressively shifts weights to correct the boundary."
            "\n- Exceptional at high-volume, sparse classification tasks."
        )
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: About the Agent
with tab_about:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("ℹ️ Project & Agent Framework")
    st.markdown(
        """
        The **Fake News Detection Agent** is a full-featured demonstration tool designed to help users identify deceptive language patterns, high-intensity sensationalism, and misleading assertions online.
        
        ##### Features:
        - **Machine Learning Core**: Employs a robust linear classifier trained on over 44,000 real and fake news articles.
        - **Rule-based NLP Audits**: Checks and highlights structural red flags (such as excessive shouting, clickbait phrases, and high emotional punctuation).
        - **Actionable Advice**: Dynamically advises users on next steps, suggesting verification sources for highly suspicious text.
        
        ##### Disclaimer
        *This prediction is generated by a Machine Learning model and should be considered as an assistance tool rather than definitive proof. Always verify important information through trusted news organizations.*
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Footer Disclaimer Section
st.markdown(
    "<p style='text-align: center; font-size: 0.8rem; color: rgba(255, 255, 255, 0.4); margin-top: 2rem; max-width: 800px; margin-left: auto; margin-right: auto;'>"
    "⚠️ <strong>Disclaimer:</strong> This prediction is generated by a Machine Learning model and should be considered as an assistance tool rather than definitive proof. Always verify important facts through trusted news organizations."
    "</p>",
    unsafe_allow_html=True
)
