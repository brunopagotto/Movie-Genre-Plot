#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[5]:


# Leitura do metadados e nomeação das colunas de acordo com a descrição do dataset
# Por se tratar de um arquivo 'tsv', é necessário definir o separador '\t'
# Como o dataset não possui nomes das colunas, define-se header = None

data_movie = pd.read_csv('movie.metadata.tsv', sep='\t', header=None)
data_movie.columns = ['Wikipedia movie ID','Freebase movie ID','Movie name','Movie release date',
                     'Movie box office revenue','Movie runtime','Movie languages','Movie countries',
                     'Movie genres']
data_movie


# In[ ]:


# Descrição dos dados

data_movie.describe

# Verificar tipos dos dados em cada coluna

data_movie.dtypes


# In[6]:


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


# In[7]:


# Exclusão das linhas onde o gênero não é citado e novo comprimento do dataframe

data_movie = data_movie[~(data_movie['Movie genres'].str.len() == 0)]

# Rearranjo dos índices de acordo com o novo tamanho do dataframe

data_movie.index = range(len(data_movie))


# In[8]:


# Criação de variável apenas com a coluna dos gêneros

genres = data_movie['Movie genres']

# Lista com os valores únicos, sem duplicações, de todos os gêneros que aparecem no dataset

genres_unique = []
for i in range(len(data_movie)):
    genres_unique = set(itertools.chain(list(genres_unique),genres[i]))
    
len(list(genres_unique))


# In[9]:


# Lista com todas as citações de gêneros no dataset
# Obs: para rodar com todas as observações, o tempo de processamento é em torno de 15 minutos,
# então, para efeito prático, utilizei apenas as 10 mil primeiras observações na criação do código, e
# ao final, rodarei com todas as observações do dataset

all_genres = sum(genres,[])
all_genres


# In[10]:


# Criar dataframe com todos os gêneros que aparecem ao menos uma vez e com a quantidade de
# aparições de cada um

all_genres_frq = nltk.FreqDist(all_genres) 
all_genres_df = pd.DataFrame({'Genre': list(all_genres_frq.keys()), 
                              'Count': list(all_genres_frq.values())})
all_genres_df


# In[11]:


# Gráfico de barra com a contagem de cada gênero por ordem, do maior para o menor, exibindo apenas
# os 30 primeiros, dos 363

g = all_genres_df.nlargest(columns = "Count", n = 30) 
plt.figure(figsize = (10,15)) 
ax = sns.barplot(data = g, x = "Count", y = "Genre") 
ax.set(ylabel = 'Genre') 
plt.show()


# In[12]:


# Leitura do arquivo texto contendo as sinopses de cada filme e seus respetivos ID's

plots = []

with open("plot_summaries.txt", 'r', encoding = "utf8") as f:
    reader = csv.reader(f, dialect = 'excel-tab') 
    for string in reader:
        plots.append(string)


# In[13]:


movie_id = []
text = []

# extract movie Ids and plot summaries
for i in plots:
    movie_id.append(i[0])
    text.append(i[1])

# create dataframe
df_plots = pd.DataFrame({'Wikipedia movie ID': movie_id, 'text': text})
df_plots = df_plots.astype({"Wikipedia movie ID": int})
df_plots


# In[14]:


df_merge = pd.merge(df_plots, data_movie[["Wikipedia movie ID", "Movie genres"]], on="Wikipedia movie ID", how="left")
df_merge


# In[15]:


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
    #text = text.split(" ")
    #text_stem = ""
    #for w in text:
    #    w = stemmer.stem(w)
    #    text_stem = text_stem + " " + w
    #text_stem = text_stem[1:]
    
    return text

for i in tqdm(range(len(df_merge))):
    df_merge.at[i,'text'] = padronizar_texto(df_merge.at[i,'text'])


# In[16]:


df_merge = df_merge.dropna()
df_merge.index = range(len(df_merge))
df_merge


# In[17]:


words = df_merge['text']

all_words = ' '.join([text for text in words])
all_words = all_words.split()

all_words_frq = nltk.FreqDist(all_words) 
words_df = pd.DataFrame({'Word':list(all_words_frq.keys()), 
                         'Count':list(all_words_frq.values())})
words_df


# In[18]:


# Gráfico de barra com a contagem de cada palavra, por ordem, do maior para o menor, 
# exibindo apenas as 50 primeiras. Lembrando que as "stopwords" foram excluídas

wf = words_df.nlargest(columns = "Count", n = 50) 
plt.figure(figsize = (10,15)) 
ax = sns.barplot(data = wf, x = "Count", y = "Word") 
ax.set(ylabel = "Word") 
plt.show()


# In[119]:


from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

# transform target variable
y = mlb.fit_transform(df_merge['Movie genres'])

tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# split dataset into training and validation set
x_train, x_test, y_train, y_test = train_test_split(df_merge['text'], y, test_size = 0.2, random_state = 42)

# Criar TF-IDF features
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)


# In[176]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Criar modelo de árvores de decisão
clf_DT = DecisionTreeClassifier(random_state = 42)

# Treinar o modelo com o conjunto de treinamento
clf_DT.fit(x_train_tfidf, y_train)

# Previsões para testar o modelo
y_pred = clf_DT.predict(x_test_tfidf)

# Utilizar o f1_score para avaliar a performance, e com o parâmetro average = "micro",
# pois as classes estão desbalanceadas nesse problema, então o objetivo é avaliar o desempenho
# geral do modelo
f1_score(y_test, y_pred, average="micro")


# In[66]:


from sklearn.ensemble import RandomForestClassifier

# Criar modelo de Random Forest
clf = RandomForestClassifier(random_state=42)

# Treinar o modelo com o conjunto de treinamento
clf.fit(x_train_tfidf, y_train)

# Previsões para testar o modelo
y_pred = clf.predict(x_test_tfidf)

# Utilizar o f1_score para avaliar a performance
f1_score(y_test, y_pred, average="micro")


# In[70]:


#!pip install xgboost
from xgboost import XGBClassifier

# Criar modelo de XGBoost
clf = XGBClassifier(random_state=42)

# Treinar o modelo com o conjunto de treinamento
clf.fit(x_train_tfidf, y_train)

# Previsões para testar o modelo
y_pred = clf.predict(x_test_tfidf)

# Utilizar o f1_score para avaliar a performance
f1_score(y_test, y_pred, average="micro")


# In[124]:


from sklearn.linear_model import LogisticRegression

# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score

lr = LogisticRegression()
clf = OneVsRestClassifier(lr)

# fit model on train data
clf.fit(x_train_tfidf, y_train)

# make predictions for validation set
y_pred = clf.predict(x_test_tfidf)

# evaluate performance
f1_score(y_test, y_pred, average="micro")


# In[65]:


# make predictions for validation set
y_pred_prob = clf.predict_proba(x_test_tfidf)

# initialize variables for storing best threshold and F1-score
best_threshold = 0
best_f1_score = 0

# loop through threshold values and calculate F1-score for each one
for threshold in np.arange(0, 1.01, 0.01):
    y_pred_threshold = (y_pred_prob > threshold).astype(int)
    f1 = f1_score(y_test, y_pred_threshold, average='micro')
    if f1 > best_f1_score:
        best_f1_score = f1
        best_threshold = threshold

print("Best threshold:", best_threshold)
print("Best F1-score:", best_f1_score)


# In[175]:


lr_updated = LogisticRegression()
clf_updated = OneVsRestClassifier(lr_updated)
clf_updated.fit(x_train_tfidf, y_train)

# make predictions for validation set using the updated clf
y_pred_prob_updated = clf_updated.predict_proba(x_test_tfidf)

# use the new threshold to predict the labels
new_y_pred = (y_pred_prob_updated >= best_threshold).astype(int)


# In[ ]:


'''

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

# transform target variable
y = df_merge['Movie genres'].apply(lambda x: ','.join(x)).tolist()
y = mlb.fit_transform(y)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# split dataset into training and validation set
x_train, x_test, y_train, y_test = train_test_split(df_merge['text'], y, test_size = 0.2, random_state = 42)

# Criar TF-IDF features
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)

# Criar modelo SVM
clf = SVC(kernel='linear')

# Treinar o modelo com o conjunto de treinamento
clf.fit(x_train_tfidf, y_train)

# Previsões para testar o modelo
y_pred = clf.predict(x_test_tfidf)

# Utilizar o f1_score para avaliar a performance
f1_score(y_test, y_pred, average="micro")

## Infelizmente, não consegui testar o modelo SVM, pois aparecia sempre que o y deveria
## ser um array unidimensional e não consegui resolver esse problema a tempo de entregar.

'''


# In[130]:


def prever_genero(entrada):
    entrada = padronizar_texto(entrada)
    entrada_tfidf = tfidf_vectorizer.transform([entrada])
    return mlb.inverse_transform(clf.predict(entrada_tfidf))


# In[131]:


sinopse = '''In 1936, archaeologist Indiana Jones braves an ancient Peruvian temple filled with booby traps to retrieve a golden idol. Upon fleeing the temple, Indiana is confronted by rival archaeologist René Belloq and the indigenous Hovitos. Surrounded and outnumbered, Indiana is forced to surrender the idol to Belloq and escapes aboard a waiting Waco seaplane, in the process revealing his fear of snakes. Shortly after returning to the college in the United States where he teaches archaeology, Indiana is interviewed by two Army intelligence agents. They inform him that the Nazis, in their quest for occult power, are searching for his old mentor, Abner Ravenwood, who is the leading expert on the ancient Egyptian city of Tanis and possesses the headpiece of an artifact called the Staff of Ra. Indiana deduces that the Nazis are searching for Tanis because it is believed to be the location of the Ark of the Covenant, the biblical chest built by the Israelites to contain the fragments of the Ten Commandments; the Nazis believe that if they acquire it, their armies will become invincible. The Staff of Ra, meanwhile, is the key to finding the Well of Souls, a secret chamber in which the Ark is buried. The agents subsequently authorize Indiana to recover the Ark before the Nazis. Indiana travels to Nepal, only to find that Ravenwood has died and that the headpiece is in the possession of his daughter, Marion, Indiana's embittered former lover. Indiana offers to buy the headpiece for three thousand dollars, plus two thousand more when they return to the United States. Marion's tavern is suddenly raided by a group of thugs commanded by Nazi agent Toht. The tavern is burned down in the ensuing fight, during which Toht burns his hand on the searing hot headpiece as he tries to grab it. Indiana and Marion escape with the headpiece, with Marion declaring she will accompany Indiana in his search for the Ark so he can repay his debt. They travel to Cairo where they learn from Indiana's friend Sallah, a skilled excavator, that Belloq and the Nazis, led by Colonel Dietrich, are currently digging for the Well of Souls with a replica of the headpiece modeled after the scar on Toht's hand. In a bazaar, Nazi operatives attempt to kidnap Marion and as Indiana chases after them it appears that she dies in an explosion. While deciphering the markings on the headpiece, Indiana and Sallah realize that the Nazis have miscalculated the location of the Well of Souls. Using this to their advantage, they infiltrate the Nazi dig and use the Staff of Ra to determine the location correctly and uncover the Well of Souls, which is filled with snakes. Indiana fends off the snakes and acquires the Ark, but Belloq, Dietrich and the Nazis arrive to take it. They toss Marion into the well with Indiana and seal them in, but they manage to escape. After a fistfight with a giant Nazi mechanic, blowing up a flying wing on the airstrip, and chasing down a convoy of trucks, Indiana takes back the Ark before it can be shipped to Berlin. Indiana and Marion leave Cairo to escort the Ark to England on board a tramp steamer. The next morning, their boat is boarded by Belloq, Dietrich and the Nazis, who once again steal the Ark and kidnap Marion. Indiana stows away on their U-boat and follows them to an isolated island in the Aegean Sea where Belloq plans to test the power of the Ark before presenting it to Hitler. Indiana reveals himself and threatens to destroy the Ark with a rocket-propelled grenade launcher, but Belloq calls his bluff, knowing Indy cannot bear to eradicate an important historical artifact. Indiana surrenders and is tied to a post with Marion as Belloq performs a ceremonial opening of the Ark, which appears to contain nothing but sand. Suddenly, spirits resembling Old Testament Seraphim emerge from the Ark. Aware of the supernatural danger of looking at the opened Ark, Indiana warns Marion to close her eyes. The apparitions suddenly morph into "angels of death", and lightning bolts begin flying out of the Ark, gruesomely killing the Nazi soldiers, while Belloq, Dietrich and Toht meet even more gruesome fates. The fires rise into the sky, then fall back down to Earth and the Ark closes with a crack of thunder. Back in Washington, D.C., the Army intelligence agents tell a suspicious Indiana and Brody that the Ark "is someplace safe" to be studied by "top men". In reality, the Ark is sealed in a wooden crate labeled "top secret" and stored in a giant government warehouse filled with countless similar crates.'''


# In[132]:


prever_genero(sinopse)


# In[ ]:


def prever_genero_DT(entrada):
    entrada = padronizar_texto(entrada)
    entrada_tfidf = tfidf_vectorizer.transform([entrada])
    return mlb.inverse_transform(clf_DT.predict(entrada_tfidf))
prever_genero_DT(sinopse)

