# Fake News Detection Agent Using Machine Learning

A complete, professional, beginner-friendly Agentic AI web application that detects whether a news article is Fake or Real using Machine Learning, calculates a confidence score, explains its classification reasoning, highlights clickbait/suspicious patterns, and maintains a local session history.

Designed as a modern, responsive, and mobile-friendly AI dashboard with glassmorphism aesthetics, styled using Bootstrap 5.

---

## Agent Workflow

Below is the step-by-step workflow followed by the AI Agent when analyzing news articles:

```
        +-----------------------------------+
        |            User Input             |
        |  (Headline or body text pasted)   |
        +-----------------------------------+
                          │
                          ▼
        +-----------------------------------+
        |        AI Agent Reads News        |
        |       (Validates input text)      |
        +-----------------------------------+
                          │
                          ▼
        +-----------------------------------+
        |           Text Cleaning           |
        |   (Lowercase, punct, stop words,  |
        |     WordNet lemmatization)        |
        +-----------------------------------+
                          │
                          ▼
        +-----------------------------------+
        |       TF-IDF Vectorization        |
        | (Transforms text to numeric data) |
        +-----------------------------------+
                          │
                          ▼
        +-----------------------------------+
        |      Machine Learning Model       |
        |  (Passive Aggressive Classifier)  |
        +-----------------------------------+
                          │
                          ▼
        +-----------------------------------+
        |            Prediction             |
        |          (Real or Fake)           |
        +-----------------------------------+
                          │
                          ▼
        +-----------------------------------+
        |         Confidence Score          |
        |   (Calculated using Sigmoid)      |
        +-----------------------------------+
                          │
                          ▼
        +-----------------------------------+
        |          AI Explanation           |
        |   (Detailed decision reasoning)   |
        +-----------------------------------+
                          │
                          ▼
        +-----------------------------------+
        |          Recommendation           |
        |  (Reputable sources verification) |
        +-----------------------------------+
```

---

## Features

- **Double-Signal Text Processing**: Merges news title and news body text for maximum context mapping.
- **Passive Aggressive Classifier**: Highly accurate (approx. 99% accuracy) text classification model.
- **Sigmoid Confidence Estimation**: Uses Platt-style sigmoid scaling to map linear distance boundaries into solid percentage scores.
- **Style Warning Flags**: Highlights sensational clickbait keywords, excessive capitalization (shouting), and extreme punctuation.
- **Session logs**: Preserves history of classified news in local cache, loading them back instantly when selected.
- **Glassmorphism UI**: Trendy, high-fidelity dark user interface styled using Bootstrap 5 and customized CSS variables.
- **Automated Presentation Generator**: Python script powered by `python-pptx` to compile a professional 9-slide PowerPoint slideshow explaining project details.

---

## Project Structure

```
FakeNewsDetectionAgent/
│
├── app.py                     # Flask Web Application Server
├── streamlit_app.py           # Streamlit AI Dashboard (suitable for Streamlit Deploy)
├── train_model.py             # Preprocessing & ML Model Training Script
├── predict.py                 # Offline CLI & Helper prediction modules
├── generate_presentation.py   # PowerPoint Slide Deck Builder script
│
├── model.pkl                  # Saved Passive Aggressive Classifier model
├── vectorizer.pkl             # Saved TF-IDF Vectorizer
├── requirements.txt           # Project dependencies definition
├── README.md                  # Comprehensive project documentation
├── presentation.pptx          # Compiled slides for project presentation
│
├── dataset/                   # Dataset Folder
│      ├── Fake.csv            # Fake news records (62.7 MB)
│      └── True.csv            # Real news records (53.5 MB)
│
├── templates/                 # HTML templates
│      └── index.html          # Dashboard user interface
│
├── static/                    # Front-end static assets
│      ├── style.css           # Glassmorphism & layout style guidelines
│      └── script.js           # AJAX operations & history controller
│
└── screenshots/               # Folder for application interface captures
```

---

## Technologies Used

- **Backend**: Python 3.11, Flask
- **Streamlit App**: Streamlit (for high-fidelity rapid AI prototyping and easy cloud deployment)
- **Frontend**: HTML5, Vanilla CSS3, Javascript (ES6), Bootstrap 5, Bootstrap Icons
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (TfidfVectorizer, PassiveAggressiveClassifier)
- **Natural Language Processing**: NLTK (Stopwords, WordNetLemmatizer)
- **Model Storage**: Joblib
- **Presentation Building**: python-pptx

---

## Installation Guide

### Prerequisites
Make sure you have **Python 3.8+** (Python 3.11 recommended) installed on your system.

### Steps
1. **Clone or copy** the `FakeNewsDetectionAgent` folder structure.
2. Open your terminal and **navigate** to the directory:
   ```bash
   cd FakeNewsDetectionAgent
   ```
3. Create a **Virtual Environment**:
   ```bash
   python -m venv venv
   ```
4. **Activate** the Virtual Environment:
   - **Windows PowerShell**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows Command Prompt (cmd)**:
     ```cmd
     .\venv\Scripts\activate.bat
     ```
   - **macOS / Linux terminal**:
     ```bash
     source venv/bin/activate
     ```
5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Execution Guide

### Step 1: Train the Machine Learning Model
Run the model training script to download NLTK data resources, process the dataset, train the Passive Aggressive Classifier, and evaluate accuracy.
```bash
python train_model.py
```
This script splits the combined datasets into an 80/20 train-test partition, prints classification reports, and outputs the saved models:
- `model.pkl`
- `vectorizer.pkl`

*(Note: You can also trigger this training pipeline directly from the Streamlit UI sidebar if model files are missing or if you want to retrain).*

### Step 2: Run the Streamlit Dashboard (Recommended for Cloud Deploy)
Launch the Streamlit app:
```bash
streamlit run streamlit_app.py
```
Once started, your browser will automatically open to:
**http://localhost:8501/**

### Step 3: Run the Flask Web Dashboard (Alternative)
Launch the Flask development server:
```bash
python app.py
```
Once started, open your browser and navigate to:
**http://127.0.0.1:5000/**

### Step 4: Run CLI Inference (Alternative)
You can test the classifier directly from your command line:
```bash
python predict.py "Breaking news: Shocking proof exposed about unverified events!"
```

### Step 5: Generate PowerPoint Presentation
Compile the project slides automatically using the python-pptx script:
```bash
python generate_presentation.py
```
This will compile a complete professional slide deck named `presentation.pptx`.

---

## Streamlit Cloud Deployment Guide

To deploy this application to **Streamlit Community Cloud**:
1. Push this workspace code to a public GitHub repository.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New app**, select your repository, branch, and specify the Main file path as `streamlit_app.py`.
4. Click **Deploy!** Streamlit will read `requirements.txt`, install all libraries (including scikit-learn, NLTK, and pandas), load the page, and automatically download the required NLTK resources to the configured local folder.
5. In the deployed dashboard, if your trained model files (`model.pkl`, `vectorizer.pkl`) are not checked in to GitHub (e.g. because they are large), you can simply click **Train Model** in the sidebar. The cloud instance will read `dataset/Fake.csv` and `dataset/True.csv`, train the model in ~1 minute, and write the model files to the instance memory for immediate interactive use!

---

## Machine Learning Workflow

1. **Preprocessing**: The combined text of titles and news is processed:
   - Punctuation, symbols, and numeric characters are removed.
   - Text is cast to lowercase.
   - Stopwords are stripped out.
   - Words are scaled back to base lemmatized roots.
2. **TF-IDF Transformation**: Texts are converted into numeric frequency matrices restricted to the top 5,000 unigram/bigram tokens.
3. **Passive Aggressive Classifier**: Passive-Aggressive algorithms update parameters if the prediction is incorrect, and remain passive if the classification is correct. This is highly suitable for large text categorization models.
4. **Sigmoid Calibration**: We map predictions to probabilities:
   $$\text{Confidence} = \frac{1}{1 + e^{-|\text{decision score}|}}$$
   This gives a percentage scale from 50% to 100%.

---

## Future Scope

- **Transformers (BERT)**: Integrating deep neural models for complex semantics.
- **Scraping APIs**: Interfacing with verification systems like Snopes, PolitiFact, or Google Fact Check API.
- **Multi-lingual analysis**: Incorporating translators to process news from multiple languages.

---

## License

Distributed under the MIT License. See `LICENSE` for more details.
