🌐 Neural Machine Translation & Sentiment Analysis

This project is an end-to-end Deep Learning and Natural Language Processing (NLP) application that combines Neural Machine Translation (NMT) with Sentiment Analysis. Users can enter English text, analyze its sentiment and emotional context, and translate it into multiple target languages through an interactive Streamlit web application.

The translation system is based on a Sequence-to-Sequence (Seq2Seq) architecture with Bahdanau Attention, while the sentiment analysis component uses a Bidirectional LSTM (BiLSTM) model to classify text into Positive, Negative, or Neutral sentiment. The application currently supports translation from English to French, Spanish, German, and Hindi.

✨ Key Features
🌍 Multi-language Translation — Translates English text into French, Spanish, German, and Hindi.
🧠 Neural Machine Translation — Uses a Seq2Seq deep learning architecture with Bahdanau Attention.
😊 Sentiment Analysis — Classifies user input as Positive, Negative, or Neutral.
🎭 Emotion Detection — Identifies the dominant emotional context of the input text.
📊 Confidence Visualization — Displays sentiment confidence and class probabilities using interactive visualizations.
📈 Translation Evaluation — Supports BLEU, ROUGE-1, ROUGE-2, ROUGE-L, and Token Accuracy metrics when a reference translation is provided.
💻 Interactive Web Interface — Provides a modern and user-friendly interface built with Streamlit.
⚡ Efficient Model Loading — Uses Streamlit caching to avoid repeatedly loading trained models.
🛠️ Technologies Used

Programming & Application

Python
Streamlit

Deep Learning

TensorFlow
Keras
Seq2Seq Architecture
Bahdanau Attention
BiLSTM

Natural Language Processing

NLTK

Data Processing & Machine Learning

NumPy
Pandas
Scikit-learn

Model Evaluation

SacreBLEU
ROUGE Score
Token Accuracy

Data Visualization

Plotly
Matplotlib
Seaborn

Development

Jupyter Notebook
IPython Kernel