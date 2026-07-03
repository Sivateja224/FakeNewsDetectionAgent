# train_model.py
# This script trains a Passive Aggressive Classifier to detect fake news.
# It cleans the text, tokenizes, removes stopwords, lemmatizes,
# applies TF-IDF vectorization, trains the model, evaluates it,
# and saves both the model and the vectorizer to disk.

import os
import re
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Step 1: Download NLTK resources
# These resources are required for text preprocessing (lemmatization and stopword removal).
print("=== Step 1: Downloading NLTK Resources ===")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print("NLTK resources downloaded successfully!\n")

# Initialize Lemmatizer and Stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and preprocesses the input text.
    1. Converts text to lowercase.
    2. Removes punctuation and numbers (keeping only letters).
    3. Tokenizes text (splits it into words).
    4. Removes stopwords (common words like 'the', 'is', 'and' that don't add semantic value).
    5. Applies lemmatization (reduces words to their base/dictionary form, e.g., 'running' -> 'run').
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase conversion
    text = text.lower()
    
    # 2. Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenize by splitting on whitespace (fast and robust)
    words = text.split()
    
    # 4 & 5. Filter out stopwords and apply lemmatization
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Reconstruct the cleaned text
    return " ".join(cleaned_words)

def main():
    # Define paths
    dataset_dir = "dataset"
    fake_path = os.path.join(dataset_dir, "Fake.csv")
    true_path = os.path.join(dataset_dir, "True.csv")
    
    # Verify dataset files exist
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print(f"Error: Dataset files not found in '{dataset_dir}' directory.")
        print("Please ensure Fake.csv and True.csv are placed in the dataset/ folder.")
        return

    # Step 2: Load datasets
    print("=== Step 2: Loading Datasets ===")
    print("Reading Fake.csv (this may take a few seconds)...")
    fake_df = pd.read_csv(fake_path)
    print(f"Loaded {len(fake_df)} fake news records.")
    
    print("Reading True.csv (this may take a few seconds)...")
    true_df = pd.read_csv(true_path)
    print(f"Loaded {len(true_df)} real news records.")
    
    # Step 3: Merge and Label datasets
    print("\n=== Step 3: Merging & Labeling Datasets ===")
    # Assign labels: Fake = 0, Real = 1
    fake_df['label'] = 0
    true_df['label'] = 1
    
    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Combined dataset contains {len(df)} total records.")
    
    # Check for empty values and drop them
    df.dropna(subset=['title', 'text'], inplace=True)
    
    # Combine Title and Text for a richer set of features
    print("Combining article titles and body text...")
    df['full_text'] = df['title'] + " " + df['text']
    
    # Step 4: Text Cleaning and Preprocessing
    print("\n=== Step 4: Preprocessing Text ===")
    print("Cleaning and lemmatizing text. This might take 1-2 minutes...")
    
    # Apply preprocessing with simple progress updates
    total_records = len(df)
    chunk_size = total_records // 10
    processed_texts = []
    
    for idx, text in enumerate(df['full_text']):
        processed_texts.append(preprocess_text(text))
        if (idx + 1) % chunk_size == 0 or idx == total_records - 1:
            percentage = int(((idx + 1) / total_records) * 100)
            print(f"Progress: {percentage}% completed ({idx + 1}/{total_records} articles)")
            
    df['cleaned_text'] = processed_texts
    print("Preprocessing completed!")
    
    # Step 5: TF-IDF Vectorization
    print("\n=== Step 5: Performing TF-IDF Vectorization ===")
    # We limit features to 5000 to keep the model size compact and avoid overfitting,
    # while allowing unigrams and bigrams (ngram_range 1 to 2) to capture context.
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    print("Fitting TF-IDF Vectorizer and transforming text...")
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label'].values
    print(f"Feature matrix shape: {X.shape}")
    
    # Step 6: Train-Test Split (80% Train, 20% Test)
    print("\n=== Step 6: Splitting Dataset (80% Train, 20% Test) ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    
    # Step 7: Train Passive Aggressive Classifier
    print("\n=== Step 7: Training Passive Aggressive Classifier ===")
    # Passive Aggressive Classifiers are excellent for text classification and large datasets.
    # They learn incrementally and adjust their weights upon encountering misclassifications.
    # Using SGDClassifier as PassiveAggressiveClassifier is deprecated in modern scikit-learn.
    pac = SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1', eta0=1.0, max_iter=50, random_state=42)
    pac.fit(X_train, y_train)
    print("Model training completed!")
    
    # Step 8: Evaluate the Model
    print("\n=== Step 8: Evaluating Model Performance ===")
    y_pred = pac.predict(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake News', 'Real News']))
    
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Step 9: Save the trained model and vectorizer
    print("\n=== Step 9: Saving Model & Vectorizer ===")
    joblib.dump(pac, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Saved model as 'model.pkl' and vectorizer as 'vectorizer.pkl' successfully!")
    print("Model training process complete!")

if __name__ == "__main__":
    main()
