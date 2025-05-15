# Ham v/s Spam Detection

This project is a machine learning-based spam detection system that classifies a given text message as either Spam or Ham (not spam). It uses a Support Vector Machine (SVM) model trained on text data, integrated with a Streamlit web interface for real-time predictions.


Live app: [Ham vs Spam](http://34.131.53.70:8502/)

### Features
- Machine learning model (SVM with linear kernel)

- Advanced text preprocessing (spelling correction, lemmatization, stemming, contractions, emoji handling, etc.)

- Trained on a labeled SMS spam dataset

- Streamlit web app for user interaction

- Model persistence using joblib

### Model Training (model.py)
#### Preprocessing Steps:

- Convert to lowercase

- Correct spelling using autocorrect

- Convert emojis to text (emoji)

- Expand contractions (contractions)

- Remove accents (textacy)

- Remove punctuation

- Tokenization (nltk)

- Stopword removal

- Stemming (SnowballStemmer)

- Lemmatization (WordNetLemmatizer)

### Pipeline:
- Custom preprocessing transformer

- CountVectorizer for feature extraction

- SVC (Support Vector Classifier) with a linear kernel

#### Output:

- Saves trained model to svm_text_pipeline.pkl

- Prints training and testing classification reports

### Web App Interface (app.py)

- Built using Streamlit

- Allows user to input a message

- Predicts and displays whether the message is Spam or Ham


### Dataset
The dataset used is hamvsspam.csv located in the ./src/ directory.

### Installation & Setup

#### 1. Clone the Repository
```bash
git clone "https://github.com/Harshita772/Ham_vs_Spam-Detection.git"
cd Ham_vs_Spam_Detection
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Download NLTK Resources
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
```

Or manually:
```bash
python -m nltk.downloader all
```

#### 4. Train the Model
```bash
python model.py
```

#### 5. Launch the Web App
```bash
streamlit run app.py
```

---

### Experiments

This project includes several Jupyter notebooks under active development to explore different modeling approaches for spam detection:

#### `BERT+SVM.ipynb`
- Combines **BERT embeddings** with a **Support Vector Machine** classifier.
- Uses pretrained BERT for feature extraction and SVM for classification.
- Useful for evaluating transformer-based embeddings on SMS spam detection.

#### `MultinomialNB.ipynb`
- Implements a **Multinomial Naive Bayes** classifier using CountVectorizer features.
- Simple yet effective baseline for text classification tasks.
- Helps compare traditional probabilistic methods with more complex models.

#### `SVM.ipynb`
- Standalone SVM model with feature extraction and evaluation.
- Mirrors the core logic used in `model.py`, providing a more interactive exploration environment.

These notebooks are intended for experimentation, model comparison, and fine-tuning. They provide insights into how different models perform on the same dataset.
