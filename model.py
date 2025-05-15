from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np
import pandas as pd
from autocorrect import Speller
from emoji import demojize
from contractions import fix
from textacy.preprocessing.remove import accents
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import nltk
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialize necessary tools
speller = Speller()
stopword = stopwords.words("english")
stem = SnowballStemmer("english")
lem = WordNetLemmatizer()

def text_pre_processing(text):
    # Lower case the text
    text = text.lower()
    # Auto Correct
    text = speller.autocorrect_sentence(text)
    # Emoji to text
    text = demojize(text)
    # Fix contractions
    text = fix(text)
    # Remove accents
    text = accents(text)
    # Remove non-alphanumeric characters (punctuation)
    text = re.sub(r"[^a-z0-9]", " ", text)

    # Tokenizing and applying stop words, stemming, and lemmatizing
    words = word_tokenize(text)
    new_text = []
    for word in words:
        if word not in stopword:
            word = stem.stem(word)
            word = lem.lemmatize(word)
            new_text.append(word)

    return " ".join(new_text)


data = pd.read_csv("./src/hamvsspam.csv", encoding='latin1')
# Keep only the first two columns (label and message)
data = data.iloc[:, :2]

# Rename the columns
data.columns = ['label', 'message']

# Apply the text preprocessing to the 'message' column
data["message"] = data["message"].apply(text_pre_processing)

label = data['label']
features = data['message']

# Split the data into training and testing sets (optional but common)
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.25, random_state=42)

# Wrap your text_pre_processing function for use in the pipeline
preprocess_transformer = FunctionTransformer(
    func=np.vectorize(text_pre_processing), validate=False
)

# Define the complete pipeline
text_clf_pipeline = Pipeline([
    ('preprocess', preprocess_transformer),
    ('vect', CountVectorizer(stop_words='english', strip_accents='unicode')),
    ('clf', SVC(kernel='linear'))
])

# Train the pipeline
text_clf_pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(text_clf_pipeline, 'svm_text_pipeline.pkl')

y_pred_train = text_clf_pipeline.predict(X_train)
y_pred_test = text_clf_pipeline.predict(X_test)


print("On training data\n",classification_report(y_train, y_pred_train))


print("On testing data\n",classification_report(y_test, y_pred_test))
