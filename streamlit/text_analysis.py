import pandas as pd
import string
from collections import Counter
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import nltk

# Baixar stopwords dinamicamente
nltk.download('stopwords')

# Função para limpar texto
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Converter para minúsculas
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remover pontuação
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remover stopwords
    return text

# Função para calcular palavras mais frequentes
def get_top_words(text_series, n=20):
    all_words = ' '.join(text_series).split()
    word_counts = Counter(all_words)
    return word_counts.most_common(n)

# Função para calcular sentimento médio
def calculate_sentiment(text_series):
    return text_series.apply(lambda x: TextBlob(x).sentiment.polarity)

# Função para gerar nuvem de palavras
def generate_wordcloud(text_series):
    text = ' '.join(text_series)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

# Função para encontrar n-gramas mais frequentes
def get_top_ngrams(text_series, ngram_range=(2, 2), n=10):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    ngrams = vectorizer.fit_transform(text_series)
    ngram_counts = ngrams.sum(axis=0)
    ngram_freq = [(word, ngram_counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    return sorted(ngram_freq, key=lambda x: x[1], reverse=True)[:n]

# Função para calcular diversidade de vocabulário
def vocabulary_diversity(text_series):
    all_words = ' '.join(text_series).split()
    unique_words = set(all_words)
    return len(unique_words) / len(all_words)

# Função para criar gráfico de barras
def plot_top_words(top_words, title):
    df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    return df
