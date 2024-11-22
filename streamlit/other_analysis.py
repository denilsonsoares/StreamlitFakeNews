import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from data_prepare import *

def distribution_by_subject(df_fake, df_true):
    df_combined = prepare_data(df_fake, df_true)
    subjects = df_combined[['id', 'subject']]
    labels = df_combined[['id', 'label']]

    # Mesclar as tabelas 'subjects' e 'labels' com base no 'id'
    data = pd.merge(subjects, labels, on='id')

    # Contar o número de notícias por assunto e por tipo (Fake ou True)
    subject_counts = data.groupby(['subject', 'label']).size().unstack().fillna(0)

    # Criar o gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))
    subject_counts.plot(kind='bar', ax=ax)
    plt.title('Distribuição de Notícias por Assunto')
    plt.xlabel('Assunto')
    plt.ylabel('Número de Notícias')
    plt.xticks(rotation=45)
    plt.legend(title='Tipo')
    st.pyplot(fig)

def temporal_analysis(df_fake, df_true):
    df_combined = prepare_data(df_fake, df_true)
    dates = df_combined[['id', 'date']]
    labels = df_combined[['id', 'label']]

    # Mesclar as tabelas 'dates' e 'labels' com base no 'id'
    data = pd.merge(dates, labels, on='id')

    # Converter a coluna 'date' para formato datetime
    data['date'] = pd.to_datetime(data['date'], errors='coerce')

    # Remover linhas com datas inválidas
    data = data.dropna(subset=['date'])

    # Extrair o ano e mês da data
    data['year_month'] = data['date'].dt.to_period('M')

    # Contar o número de notícias por tipo e por período
    count_data = data.groupby(['year_month', 'label']).size().unstack(fill_value=0)

    # Plotar a quantidade de notícias ao longo do tempo
    fig, ax = plt.subplots(figsize=(12, 6))
    if 'Fake' in count_data.columns:
        count_data['Fake'].plot(kind='line', marker='o', color='red', label='Fake News', ax=ax)
    if 'True' in count_data.columns:
        count_data['True'].plot(kind='line', marker='o', color='blue', label='True News', ax=ax)

    plt.title('Quantidade de Notícias Publicadas ao Longo do Tempo')
    plt.xlabel('Ano e Mês')
    plt.ylabel('Quantidade de Notícias')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

def text_length_analysis(df_fake, df_true):
    df_combined = prepare_data(df_fake, df_true)
    articles = df_combined[['id', 'text']]
    labels = df_combined[['id', 'label']]

    # Mesclar as tabelas 'articles' e 'labels'
    data = pd.merge(articles, labels, on='id')

    # Calcular o comprimento do texto
    data['text_length'] = data['text'].str.len()

    # Calcular o comprimento médio do texto por tipo de notícia
    avg_text_length = data.groupby('label')['text_length'].mean()

    # Criar o gráfico de barras com tamanho menor
    fig, ax = plt.subplots(figsize=(4, 3))
    avg_text_length.plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
    plt.title('Comprimento Médio do Texto', fontsize=8)
    plt.xlabel('Tipo de Notícia', fontsize=5)
    plt.ylabel('Comprimento Médio', fontsize=5)
    plt.xticks(rotation=0, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

def subject_text_length_correlation(df_fake, df_true):
    df_combined = prepare_data(df_fake, df_true)
    articles = df_combined[['id', 'text']]
    subjects = df_combined[['id', 'subject']]
    labels = df_combined[['id', 'label']]

    # Mesclar 'articles', 'subjects', e 'labels'
    data = pd.merge(articles, subjects, on='id')
    data = pd.merge(data, labels, on='id')

    # Calcular o comprimento do texto
    data['text_length'] = data['text'].str.len()

    # Calcular o comprimento médio do texto por assunto
    subject_text_length = data.groupby('subject')['text_length'].mean()

    # Criar o gráfico de barras horizontais
    fig, ax = plt.subplots(figsize=(8, 6))
    subject_text_length.sort_values().plot(kind='barh', ax=ax)
    plt.title('Comprimento Médio do Texto por Assunto')
    plt.xlabel('Comprimento Médio do Texto')
    plt.ylabel('Assunto')
    st.pyplot(fig)

def correlation_matrix_analysis(df_fake, df_true):
    df_combined = prepare_data(df_fake, df_true)
    labels = df_combined[['id', 'label']]
    subjects = df_combined[['id', 'subject']]

    # Mesclar as tabelas 'labels' e 'subjects' com base no 'id'
    data = pd.merge(labels, subjects, on='id')

    # Codificar o tipo de notícia como 0 (Fake) e 1 (True)
    data['label_numeric'] = data['label'].map({'Fake': 0, 'True': 1})

    # Aplicar One-Hot Encoding nos assuntos
    subjects_encoded = pd.get_dummies(data['subject'])

    # Combinar os dados de tipo de notícia com os assuntos
    correlation_data = pd.concat([data[['label_numeric']], subjects_encoded], axis=1)

    # Calcular a correlação apenas entre 'label_numeric' e os assuntos
    correlation_with_label = correlation_data.corr()['label_numeric'].drop('label_numeric')

    # Plotar a correlação
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_with_label.sort_values().plot(kind='barh', color='skyblue', ax=ax)
    plt.title('Correlação entre Tipo de Notícia (Fake/True) e Assuntos')
    plt.xlabel('Correlação')
    plt.ylabel('Assunto')
    plt.grid(axis='x')
    st.pyplot(fig)
