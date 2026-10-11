import pandas as pd
import numpy as np
import sklearn
import nltk
import streamlit as st
import string
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# ============================================================
# 1. ENVIRONMENT TEST
# ============================================================

print("--- AI Spam Message Detector Environment Test ---")
print("Pandas version:", pd.__version__)
print("Numpy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("NLTK version:", nltk.__version__)
print("Streamlit version:", st.__version__)
print("-------------------------------------------------")
print("All libraries imported successfully! Virtual Environment is working perfectly.")


# ============================================================
# 2. LOAD THE DATASET
# ============================================================

DATASET_PATH = "dataset/spam.csv"

print("\n--- LOADING DATASET ---")

data = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")


# ============================================================
# 3. BASIC DATASET INSPECTION
# ============================================================

print("\n--- FIRST 5 ROWS ---")
print(data.head())

print("\n--- DATA SHAPE ---")
print(data.shape)

print("\n--- COLUMN NAMES ---")
print(data.columns)

print("\n--- DATA SUMMARY (INFO) ---")
data.info()

print("\n--- CLASS DISTRIBUTION ---")
print(data["label"].value_counts())


# ============================================================
# 4. DUPLICATE CHECK
# ============================================================

print("\n--- DUPLICATES CHECK ---")

duplicate_count = data.duplicated().sum()

print("Total Duplicate Rows:", duplicate_count)

if duplicate_count > 0:
    print(f"Removing {duplicate_count} duplicates...")
    data = data.drop_duplicates(keep="first")
    print("New Shape after removing duplicates:", data.shape)
else:
    print("No duplicate rows found.")


# ============================================================
# 5. LABEL ENCODING
# ============================================================

print("\n--- LABEL ENCODING ---")

print("Unique labels before encoding:", data["label"].unique())

# ham -> 0
# spam -> 1
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

print("Unique labels after encoding:", data["label"].unique())

print("Encoded Class Distribution:")
print(data["label"].value_counts())


# ============================================================
# 6. BASIC TEXT CLEANING FUNCTION
# ============================================================

def basic_text_cleaning(text):
    """
    Basic text preprocessing:
    1. Convert text to lowercase
    2. Remove punctuation
    3. Remove extra whitespaces
    """

    text = str(text)

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # 3. Remove extra whitespaces
    text = " ".join(text.split())

    return text


# ============================================================
# 7. TEXT CLEANING DEMO
# ============================================================

print("\n--- TEXT CLEANING DEMO ---")

sample_msg = "WINNER!! Claim your free reward NOW... Call at 90812 !!!"

print("Original:", sample_msg)
print("Cleaned :", basic_text_cleaning(sample_msg))


# ============================================================
# 8. APPLY CLEANING TO DATASET
# ============================================================

print("\n--- APPLYING TEXT CLEANING ---")

data["cleaned_message"] = data["message"].apply(
    basic_text_cleaning
)

print("\n--- SAMPLE CLEANED DATASET ---")

print(
    data[["message", "cleaned_message"]].head()
)


# ============================================================
# 9. CREATE ML-READY DATASET
# ============================================================

cleaned_data = data[
    ["label", "cleaned_message"]
].copy()

cleaned_data = cleaned_data.rename(
    columns={
        "cleaned_message": "message"
    }
)

print("\n--- ML-READY DATASET PREVIEW ---")

print(cleaned_data.head())

print(
    "Cleaned Dataset Shape:",
    cleaned_data.shape
)


# ============================================================
# 10. SAVE CLEANED DATASET
# ============================================================

CLEANED_DATASET_PATH = "dataset/cleaned_spam.csv"

cleaned_data.to_csv(
    CLEANED_DATASET_PATH,
    index=False
)

print(
    f"\nCleaned dataset successfully saved to "
    f"'{CLEANED_DATASET_PATH}'!"
)


# ============================================================
# 11. CLASS DISTRIBUTION VISUALIZATION
# ============================================================

print("\n--- GENERATING DATA VISUALIZATION ---")

class_counts = cleaned_data["label"].value_counts()

plt.figure(figsize=(6, 4))

class_counts.plot(kind="bar")

plt.title("Distribution of Ham vs Spam Messages")
plt.xlabel("Class (0 = Ham, 1 = Spam)")
plt.ylabel("Number of Messages")
plt.xticks(rotation=0)

plot_path = "dataset/class_distribution.png"

plt.tight_layout()
plt.savefig(plot_path)
plt.close()

print(
    f"Visualization saved successfully as "
    f"'{plot_path}'!"
)


# ============================================================
# 12. BAG OF WORDS DEMO
# ============================================================

print("\n--- BAG OF WORDS DEMO ---")

# Tiny sample corpus
tiny_corpus = [
    "I love Python",
    "I love AI"
]

# Initialize CountVectorizer
bow_vectorizer = CountVectorizer()

# Fit and transform the corpus
X_tiny_bow = bow_vectorizer.fit_transform(
    tiny_corpus
)

# Print vocabulary
print(
    "Vocabulary:",
    bow_vectorizer.vocabulary_
)

# Print feature names
print(
    "Feature Names:",
    bow_vectorizer.get_feature_names_out()
)

# Convert sparse matrix to readable array
print(
    "Numerical Matrix:\n",
    X_tiny_bow.toarray()
)


# ============================================================
# 13. LOAD CLEANED DATASET FOR ML
# ============================================================

print("\n--- LOADING CLEANED DATASET FOR ML ---")

cleaned_data = pd.read_csv(
    CLEANED_DATASET_PATH
)

# Handle missing rows
cleaned_data = cleaned_data.dropna(
    subset=["message", "label"]
)

X = cleaned_data["message"]
y = cleaned_data["label"]


# ============================================================
# 14. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- SPLIT DATA SHAPES ---")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 15. BASIC TF-IDF
# ============================================================

print("\n--- BASIC TF-IDF ---")

tfidf_vectorizer = TfidfVectorizer()

# IMPORTANT:
# Fit ONLY on training data
X_train_tfidf = tfidf_vectorizer.fit_transform(
    X_train
)

# IMPORTANT:
# Only transform test data
X_test_tfidf = tfidf_vectorizer.transform(
    X_test
)

print("\n--- TF-IDF MATRIX SHAPES ---")

print(
    "X_train_tfidf shape:",
    X_train_tfidf.shape
)

print(
    "X_test_tfidf shape:",
    X_test_tfidf.shape
)


# ============================================================
# 16. CUSTOMIZED TF-IDF
# ============================================================

print("\n--- CUSTOMIZED TF-IDF ---")

custom_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000,
    ngram_range=(1, 2)
)

# Fit and transform ONLY on training data
X_train_custom = custom_vectorizer.fit_transform(
    X_train
)

# Transform test data using SAME fitted vectorizer
X_test_custom = custom_vectorizer.transform(
    X_test
)

print("\n--- CUSTOMIZED TF-IDF MATRIX SHAPES ---")

print(
    "Custom X_train_custom shape:",
    X_train_custom.shape
)

print(
    "Custom X_test_custom shape:",
    X_test_custom.shape
)


# ============================================================
# 17. INSPECT GENERATED FEATURES
# ============================================================

feature_names = (
    custom_vectorizer
    .get_feature_names_out()
)

print("\n--- SAMPLE FEATURE NAMES ---")

print("First 20 features:")

print(
    feature_names[:20]
)


# ============================================================
# 18. DATA LEAKAGE CHECK / CONCEPT
# ============================================================

print("\n--- DATA LEAKAGE CHECK ---")

print(
    "Correct approach used:"
)

print(
    "Training data -> fit_transform()"
)

print(
    "Testing data  -> transform()"
)

print(
    "The TF-IDF vocabulary and IDF statistics "
    "are learned only from the training data."
)

print(
    "\n--- PHASE 3 COMPLETED ---"
)

print(
    "Dataset -> Cleaning -> BoW -> Train/Test Split "
    "-> TF-IDF -> Unigram + Bigram features"
)


# ============================================================
# 19. MODEL TRAINING & PREDICTION
# ============================================================

print("\n--- MODEL TRAINING ---")

# Initialize models
nb_model = MultinomialNB()
lr_model = LogisticRegression(max_iter=1000)
svm_model = LinearSVC(random_state=42)

# Train models on customized training features
nb_model.fit(X_train_custom, y_train)
print("Naive Bayes trained successfully.")

lr_model.fit(X_train_custom, y_train)
print("Logistic Regression trained successfully.")

svm_model.fit(X_train_custom, y_train)
print("Linear SVM trained successfully.")

# Generate predictions on test data
nb_preds = nb_model.predict(X_test_custom)
lr_preds = lr_model.predict(X_test_custom)
svm_preds = svm_model.predict(X_test_custom)

print("\nPredictions generated for all models!")


# ============================================================
# 20. TEST PREDICTIONS ON NEW SAMPLES
# ============================================================

sample_messages = [
    "Congratulations! You won a free prize!",
    "Hey, are you coming to college today?"
]

# Preprocess and vectorize the sample messages
cleaned_samples = [basic_text_cleaning(msg) for msg in sample_messages]
vectorized_samples = custom_vectorizer.transform(cleaned_samples)

print("\n--- TEST PREDICTIONS ON NEW SAMPLES ---")
for msg, vec_msg in zip(sample_messages, vectorized_samples):
    nb_pred = nb_model.predict(vec_msg)[0]
    lr_pred = lr_model.predict(vec_msg)[0]
    svm_pred = svm_model.predict(vec_msg)[0]
    
    print(f"\nMessage: '{msg}'")
    print(f"  Naive Bayes Prediction        : {'Spam' if nb_pred == 1 else 'Ham'}")
    print(f"  Logistic Regression Prediction: {'Spam' if lr_pred == 1 else 'Ham'}")
    print(f"  Linear SVM Prediction         : {'Spam' if svm_pred == 1 else 'Ham'}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import joblib

# 1. Evaluate performance of each model
models_performance = {}

for name, preds in [("Naive Bayes", nb_preds), ("Logistic Regression", lr_preds), ("Linear SVM", svm_preds)]:
    models_performance[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1-score": f1_score(y_test, preds)
    }

# Display Comparison Table
print("\n--- MODEL COMPARISON TABLE ---")
print(f"{'Model':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-score':<8}")
print("-" * 63)
for model_name, metrics in models_performance.items():
    print(f"{model_name:<22} | {metrics['Accuracy']:0.6f} | {metrics['Precision']:0.6f} | {metrics['Recall']:0.6f} | {metrics['F1-score']:0.6f}")


# 2. Select the best model (We prioritize high Precision to avoid marking safe mails as spam, and high F1-score)
best_model_name = "Naive Bayes"
best_precision = models_performance[best_model_name]["Precision"]
best_model = nb_model

for name, model_obj in [("Logistic Regression", lr_model), ("Linear SVM", svm_model)]:
    # Select based on Precision first
    if models_performance[name]["Precision"] > best_precision:
        best_precision = models_performance[name]["Precision"]
        best_model_name = name
        best_model = model_obj

print(f"\n---> Best Model Selected based on Precision: {best_model_name} (Precision: {best_precision:0.6f})")


# 3. Save the best model and custom vectorizer to the 'model/' directory
os.makedirs("model", exist_ok=True)

model_path = "model/spam_model.pkl"
vectorizer_path = "model/tfidf_vectorizer.pkl"

joblib.dump(best_model, model_path)
joblib.dump(custom_vectorizer, vectorizer_path)

print(f"\nSaved Best Model successfully to '{model_path}'!")
print(f"Saved TF-IDF Vectorizer successfully to '{vectorizer_path}'!")

