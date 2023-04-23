#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Bibliotecas que serão usadas

import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import itertools
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[3]:


# Leitura do arquivo texto contendo as sinopses de cada filme e seus respetivos ID's

plots = []

with open("plot_summaries.txt", 'r', encoding = "utf8") as f:
    reader = csv.reader(f, dialect = 'excel-tab') 
    for string in reader:
        plots.append(string)


# In[4]:


movie_id = []
text = []

# extract movie Ids and plot summaries
for i in plots:
    movie_id.append(i[0])
    text.append(i[1])

# create dataframe
df_plots = pd.DataFrame({'Wikipedia movie ID': movie_id, 'text': text})
df_plots


# In[5]:


df_plots.dtypes


# In[6]:


df_plots = df_plots.astype({"Wikipedia movie ID": int})


# In[7]:


# Leitura do metadados e nomeação das colunas de acordo com a descrição do dataset
# Por se tratar de um arquivo 'tsv', é necessário definir o separador '\t'
# Como o dataset não possui nomes das colunas, define-se header = None

data_movie = pd.read_csv('movie.metadata.tsv',sep='\t',header=None)
data_movie.columns = ['Wikipedia movie ID','Freebase movie ID','Movie name','Movie release date',
                     'Movie box office revenue','Movie runtime','Movie languages','Movie countries',
                     'Movie genres']
data_movie


# In[8]:


# Descrição dos dados

data_movie.describe


# In[9]:


# Função para extrair das colunas apenas informação relevante, excluindo códigos e pontuações

colunas = ['Movie languages','Movie countries','Movie genres']

def reduzir_string(col_name):
    for i in range(len(data_movie)):
        list_genres = data_movie.at[i,col_name][1:-2].replace('": "','", "').split('", "')
        # a função replace foi usada para padronizar a separação dos caracteres
        # a função split foi usada para separar os gêneros dentro de um vetor
        list_genres = list_genres[1::2]
        # como, no dataset, alterna-se código e gênero, essa etapa seleciona apenas os elementos
        # ímpares dentro do vetor (1,3,5,...)
        data_movie.at[i,col_name] = list_genres
        # com as modificações feitas, substitui-se na coluna de interesse o resultado desejado
        
for col in colunas:
        reduzir_string(col)

data_movie


# In[10]:


# Verificar tipos dos dados em cada coluna

data_movie.dtypes


# In[11]:


# Comprimento de linhas do dataframe

len(data_movie)


# In[12]:


# Número de ocorrências onde não há registro de gênero do filme

len(data_movie[data_movie['Movie genres'].str.len() == 0])


# In[13]:


# Exclusão das linhas onde o gênero não é citado e novo comprimento do dataframe

data_movie = data_movie[~(data_movie['Movie genres'].str.len() == 0)]
len(data_movie)


# In[14]:


# Rearranjo dos índices de acordo com o novo tamanho do dataframe

data_movie.index = range(len(data_movie))
data_movie.index


# In[15]:


data_movie.iloc[25]


# In[16]:


# Criação de variável apenas com a coluna dos gêneros

genres = data_movie['Movie genres']
genres


# In[17]:


# Lista com os valores únicos, sem duplicações, de todos os gêneros que aparecem no dataset

genres_unique = []
for i in range(len(data_movie)):
    genres_unique = set(itertools.chain(list(genres_unique),genres[i]))
list(genres_unique)


# In[18]:


# Número de gêneros que aparecem no dataset e serão parte da variável de interesse

len(list(genres_unique))


# In[26]:


# Lista com todas as citações de gêneros no dataset
# Obs: para rodar com todas as observações, o tempo de processamento é em torno de 15 minutos,
# então, para efeito prático, utilizei apenas as 10 mil primeiras observações na criação do código, e
# ao final, rodarei com todas as observações do dataset

all_genres = sum(genres,[])
all_genres

#ab = []
#for i in tqdm(range(len(data_movie))):
#    ab = itertools.chain(list(ab),genres[i])
#    
#list(ab)


# In[27]:


len(all_genres)


# In[30]:


# Comprimento da lista com todos as citações de gêneros
len(all_genres)


# In[29]:


# Criar dataframe com todos os gêneros que aparecem ao menos uma vez e com a quantidade de
# aparições de cada um

all_genres_frq = nltk.FreqDist(all_genres) 
all_genres_df = pd.DataFrame({'Genre': list(all_genres_frq.keys()), 
                              'Count': list(all_genres_frq.values())})
all_genres_df


# In[31]:


# Gráfico de barra com a contagem de cada gênero por ordem, do maior para o menor, exibindo apenas
# os 30 primeiros, dos 363

g = all_genres_df.nlargest(columns = "Count", n = 30) 
plt.figure(figsize = (10,15)) 
ax = sns.barplot(data = g, x = "Count", y = "Genre") 
ax.set(ylabel = 'Genre') 
plt.show()


# In[ ]:


countries = data_movie['Movie countries']
countries


# In[ ]:


i = 0
one_country = [countries[i] for i in range(len(countries)) if len(countries[i]) == 1]
one_country


# In[ ]:


len(one_country)


# In[26]:


all_one_country = sum(one_country[0:10000],[])
all_one_country


# In[27]:


# Criar dataframe com todos os países que produziram filmes sem coparticipação e com a quantidade de
# aparições de cada um

all_one_country = nltk.FreqDist(all_one_country) 
all_one_country_df = pd.DataFrame({'Country': list(all_one_country.keys()), 
                              'Count': list(all_one_country.values())})
all_one_country_df


# In[28]:


# Gráfico de barra com a contagem de cada país em filmes sem coparticipação de países,
# por ordem, do maior para o menor, exibindo apenas os 10 primeiros

g = all_one_country_df.nlargest(columns = "Count", n = 10) 
plt.figure(figsize = (10,15)) 
ax = sns.barplot(data = g, x = "Count", y = "Country") 
ax.set(ylabel = 'Country') 
plt.show()


# In[29]:


i = 0
two_more_countries = [countries[i] for i in range(len(countries)) if len(countries[i]) > 1]
two_more_countries


# In[30]:


len(two_more_countries)


# In[31]:


all_two_more_countries = sum(two_more_countries[0:10000],[])
all_two_more_countries


# In[32]:


# Criar dataframe com todos os países que produziram filmes em conjunto e com a quantidade de
# aparições de cada um

all_two_more_countries = nltk.FreqDist(all_two_more_countries) 
all_two_more_countries_df = pd.DataFrame({'Country': list(all_two_more_countries.keys()), 
                              'Count': list(all_two_more_countries.values())})
all_two_more_countries_df


# In[33]:


# Gráfico de barra com a contagem de cada país em filmes com participação de dois ou mais países,
# por ordem, do maior para o menor, exibindo apenas os 10 primeiros

g = all_two_more_countries_df.nlargest(columns = "Count", n = 10) 
plt.figure(figsize = (10,15)) 
ax = sns.barplot(data = g, x = "Count", y = "Country") 
ax.set(ylabel = 'Country') 
plt.show()


# In[34]:


languages = data_movie['Movie languages']
languages


# In[35]:


i = 0
one_language = [languages[i] for i in range(len(languages)) if len(languages[i]) == 1]
one_language


# In[36]:


len(one_language)


# In[37]:


all_one_language = sum(one_language[0:10000],[])
all_one_language


# In[38]:


# Criar dataframe com todos os países que produziram filmes em conjunto e com a quantidade de
# aparições de cada um

all_one_language = nltk.FreqDist(all_one_language) 
all_one_language_df = pd.DataFrame({'Language': list(all_one_language.keys()), 
                              'Count': list(all_one_language.values())})
all_one_language_df


# In[39]:


# Gráfico de barra com a contagem de cada país em filmes com participação de dois ou mais países,
# por ordem, do maior para o menor, exibindo apenas os 10 primeiros

g = all_one_language_df.nlargest(columns = "Count", n = 10) 
plt.figure(figsize = (10,15)) 
ax = sns.barplot(data = g, x = "Count", y = "Language") 
ax.set(ylabel = 'Language') 
plt.show()


# In[40]:


i = 0
two_more_languages = [languages[i] for i in range(len(languages)) if len(languages[i]) > 1]
two_more_languages


# In[41]:


len(two_more_languages)


# In[42]:


all_two_more_languages = sum(two_more_languages[0:10000],[])
all_two_more_languages


# In[43]:


# Criar dataframe com todos os países que produziram filmes em conjunto e com a quantidade de
# aparições de cada um

all_two_more_languages = nltk.FreqDist(all_two_more_languages) 
all_two_more_languages_df = pd.DataFrame({'Language': list(all_two_more_languages.keys()), 
                              'Count': list(all_two_more_languages.values())})
all_two_more_languages_df


# In[44]:


# Gráfico de barra com a contagem de cada país em filmes com participação de dois ou mais países,
# por ordem, do maior para o menor, exibindo apenas os 10 primeiros

g = all_two_more_languages_df.nlargest(columns = "Count", n = 10) 
plt.figure(figsize = (10,15)) 
ax = sns.barplot(data = g, x = "Count", y = "Language") 
ax.set(ylabel = 'Language') 
plt.show()


# In[45]:


# Alguns pontos interessantes, a partir desses gráficos com os países de origem de cada filme
# e seus idiomas, são a diferença entre o número de filmes produzidos exclusivamente na Índia e
# outros países asiáticos, se comparados esses mesmos países em produções com coparticipação
# de países, e o grande número de filmes com a categoria "Silent film" quando há mais de um 
# idioma presente no filme.

# Como o objetivo da avaliação está na criação de um modelo para prever os gêneros dos filmes a
# partir de suas sinopses, não continuarei a análise exploratória dos outros dados. Daqui em diante,
# utilizarei um banco de dados apenas com as variáveis relevantes para o problema, que são as 
# variáveis contendo o ID e o gênero do filme, no arquivo metadados, e as variáveis contendo o ID
# e a sinopse no arquivo plot_summaries


# In[46]:


data_movie


# In[33]:


df_merge = pd.merge(df_plots, data_movie[["Wikipedia movie ID", "Movie genres"]], on="Wikipedia movie ID", how="left")
df_merge


# In[48]:


df_merge.at[5,'text']


# In[34]:


# Função para limpar o texto: padronizando com letra minúscula, removendo espaços e caracteres
# que não fazem parte de palavras, removendo as "stopwords", que são palavras que não agregam
# ao conteúdo, apenas à forma, do texto, e utilizando a técnica Stemming, que reduz as palavras
# ao seu radical

stop = stopwords.words('english')
stemmer = PorterStemmer()

def padronizar_texto(text):
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub("[^a-zA-Z]"," ",text)
    text = " ".join([word for word in text.split() if word not in (stop)])
    text = text.split(" ")
    text_stem = ""
    for w in text:
        w = stemmer.stem(w)
        text_stem = text_stem + " " + w
    text_stem = text_stem[1:]
    
    return text_stem


# In[50]:


padronizar_texto(df_merge.at[42299,'text'])


# In[35]:


#for i in tqdm(range(len(df_merge))):
#    df_merge.at[i,'text'] = padronizar_texto(df_merge.at[i,'text'])
    
for i in tqdm(range(len(df_merge))):
    df_merge.at[i,'text'] = padronizar_texto(df_merge.at[i,'text'])


# In[36]:


df_merge


# In[37]:


df_merge_backup = df_merge.copy()


# In[38]:


#df_merge_backup = df_merge_backup.dropna()
#df_merge_backup.index = range(len(df_merge_backup))
df_merge_backup


# In[39]:


df_merge = df_merge.dropna()
df_merge.index = range(len(df_merge))
df_merge


# In[40]:


words = df_merge['text']
words


# In[41]:


all_words = ' '.join([text for text in words])
all_words = all_words.split()


# In[42]:


all_words_frq = nltk.FreqDist(all_words) 
words_df = pd.DataFrame({'Word':list(all_words_frq.keys()), 
                         'Count':list(all_words_frq.values())})
words_df


# In[43]:


# Gráfico de barra com a contagem de cada palavra, por ordem, do maior para o menor, 
# exibindo apenas as 50 primeiras. Lembrando que as "stopwords" foram excluídas

wf = words_df.nlargest(columns = "Count", n = 50) 
plt.figure(figsize = (10,15)) 
ax = sns.barplot(data = wf, x = "Count", y = "Word") 
ax.set(ylabel = "Word") 
plt.show()


# In[44]:


df_merge.dtypes


# In[45]:


df_merge['Movie genres']


# In[60]:


from sklearn.preprocessing import MultiLabelBinarizer

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(df_merge['Movie genres'])

# transform target variable
y = multilabel_binarizer.transform(df_merge['Movie genres'])

# transform target variable
#y = mlb.fit_transform(df_merge['Movie genres'])

#
#classes = mlb.classes_
#print(classes)


# In[63]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)


# In[64]:


# split dataset into training and validation set
xtrain, xval, ytrain, yval = train_test_split(df_merge['text'], y, test_size=0.2, random_state=9)


# In[65]:


# create TF-IDF features
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)


# In[68]:


from sklearn.linear_model import LogisticRegression

# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score


# In[69]:


lr = LogisticRegression()
clf = OneVsRestClassifier(lr)


# In[70]:


# fit model on train data
clf.fit(xtrain_tfidf, ytrain)


# In[71]:


# make predictions for validation set
y_pred = clf.predict(xval_tfidf)


# In[72]:


y_pred[3]


# In[81]:


multilabel_binarizer.inverse_transform(y_pred)[20]


# In[74]:


# evaluate performance
f1_score(yval, y_pred, average="micro")


# In[75]:


# predict probabilities
y_pred_prob = clf.predict_proba(xval_tfidf)


# In[76]:


t = 0.3 # threshold value
y_pred_new = (y_pred_prob >= t).astype(int)


# In[77]:


# evaluate performance
f1_score(yval, y_pred_new, average="micro")


# In[48]:


## ChatGPT

from sklearn.feature_extraction.text import CountVectorizer

# Criar uma instância do CountVectorizer
vectorizer = CountVectorizer()

# Aplicar o CountVectorizer aos dados de sinopse
features = vectorizer.fit_transform(df_merge['text'])


# In[59]:


len(y)


# In[49]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=25)

# Criar uma instância do modelo de regressão logística
model = LogisticRegression()

# Treinar o modelo usando os dados de treinamento
model.fit(X_train, y_train)

# Avaliar o modelo usando os dados de teste
score = model.score(X_test, y_test)


# In[295]:


import nltk
nltk.download('punkt')  # necessário para tokenizar as sinopses
nltk.download('stopwords')  # necessário para remover palavras comuns sem significado

from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[305]:


nltk.download('movie_reviews')  # necessário para remover palavras comuns sem significado


# In[ ]:


# cria um dicionário com as palavras mais frequentes nas resenhas de filmes
words_reviews = nltk.FreqDist(w.lower() for w in movie_reviews.words() if w.lower() not in stopwords.words('english'))


# In[302]:


# seleciona as 2000 palavras mais frequentes para utilizar como features
word_features = list(words_reviews)[:2000]


# In[ ]:





# In[ ]:


# função que extrai as features de uma sinopse
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# carrega as sinopses dos filmes e seus respectivos gêneros
documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

# cria as features para cada sinopse
featuresets = [(document_features(d), c) for (d,c) in documents]

# divide as sinopses em dois conjuntos: treinamento (80%) e teste (20%)
train_set, test_set = featuresets[:1600], featuresets[1600:]

# treina o classificador Naive Bayes com o conjunto de treinamento
classifier = nltk.NaiveBayesClassifier.train(train_set)

# utiliza o classificador para classificar as sinopses de teste
for document, category in test_set:
    genre = classifier.classify(document)
    print("Sinopse: ", " ".join(document)[:50], "...")
    print("Gênero estimado: ", genre)
    print("Gênero real: ", category)
    print("\n")


# In[285]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)


# In[286]:



# split dataset into training and validation set
x_train, x_test, y_train, y_test = train_test_split(df_merge['Movie genres'], 
                                                    y, 
                                                    test_size=0.25, 
                                                    random_state=25)


# In[290]:


# create TF-IDF features
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)


# In[291]:


x_test_tfidf = tfidf_vectorizer.transform(x_test)


# In[ ]:




