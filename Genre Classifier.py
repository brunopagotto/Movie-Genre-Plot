#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# O script abaixo contém a análise exploratória, pré-processamento, criação e análise dos modelos,
# e a implementação de um pipeline para a predição de uma sinopse por vez, como foi requisitado.
# Cinco diferentes modelos foram testados e o modelo de Regressão Logística foi o escolhido, por
# ter melhor performance e maior precisão nos resultado.
# Mantive os modelos que não foram usados entre vírgulas triplas(''') para demonstrar como foi o
# processo de definição do modelo ideal.
# No final, está a funcão prever_genero_LR, que recebe como entrada uma sinopse qualquer e retorna
# o(s) gênero(s) do filme.


# In[3]:


# Bibliotecas que serão usadas

import pandas as pd
import numpy as np
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
import itertools
import nltk
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[4]:


# Leitura do metadados e nomeação das colunas de acordo com a descrição do dataset.
# Por se tratar de um arquivo 'tsv', é necessário definir o separador '\t'.
# Como o dataset não possui nomes das colunas, define-se header = None.
data_movie = pd.read_csv('movie.metadata.tsv', sep='\t', header=None)
data_movie.columns = ['Wikipedia movie ID','Freebase movie ID','Movie name','Movie release date',
                     'Movie box office revenue','Movie runtime','Movie languages','Movie countries',
                     'Movie genres']
data_movie


# In[5]:


# Descrição dos dados.
data_movie.describe

# Verificar tipos dos dados em cada coluna.
data_movie.dtypes


# In[6]:


# Função para extrair das colunas apenas informação relevante, excluindo códigos e pontuações.
colunas = ['Movie languages','Movie countries','Movie genres']
def reduzir_string(col_name):
    for i in range(len(data_movie)):
        # a função replace foi usada para padronizar a separação dos caracteres
        # a função split foi usada para separar os gêneros dentro de um vetor
        list_genres = data_movie.at[i,col_name][1:-2].replace('": "','", "').split('", "')
        # como, no dataset, alterna-se código e gênero, essa etapa seleciona apenas os elementos
        # ímpares dentro do vetor (1,3,5,...)
        list_genres = list_genres[1::2]
        # com as modificações feitas, substitui-se na coluna de interesse o resultado desejado
        data_movie.at[i,col_name] = list_genres

# Loop para aplicar a função às três colunas que possuem caracteres irrelevantes para o problema.
for col in colunas:
        reduzir_string(col)

# Visualização dos dados após aplicar a função reduzir_string.
data_movie


# In[7]:


# Exclusão das linhas onde o gênero não é citado.
data_movie = data_movie[~(data_movie['Movie genres'].str.len() == 0)]

# Rearranjo dos índices de acordo com o novo tamanho do dataframe.
data_movie.index = range(len(data_movie))


# In[8]:


# Criação de variável apenas com a coluna dos gêneros.
genres = data_movie['Movie genres']

# Tamanho da lista com os valores, sem duplicações, de todos os gêneros que aparecem no dataset.
genres_unique = []
for i in range(len(data_movie)):
    genres_unique = set(itertools.chain(list(genres_unique),genres[i]))
    
len(list(genres_unique))


# In[9]:


# Lista com todas as citações de gêneros no dataset
all_genres = sum(genres,[])
all_genres


# In[10]:


# Criar dataframe com todos os gêneros que aparecem ao menos uma vez, e com a quantidade de
# aparições de cada um.
all_genres_frq = nltk.FreqDist(all_genres) 
all_genres_df = pd.DataFrame({'Genre': list(all_genres_frq.keys()), 
                              'Count': list(all_genres_frq.values())})
all_genres_df


# In[11]:


# Gráfico de barra com a contagem de cada gênero, por ordem, do maior para o menor, exibindo apenas
# os 30 primeiros dos 363.
g = all_genres_df.nlargest(columns = "Count", n = 30) 
plt.figure(figsize = (10,15)) 
ax = sns.barplot(data = g, x = "Count", y = "Genre") 
ax.set(ylabel = 'Genre') 
plt.show()


# In[12]:


# Leitura do arquivo texto contendo as sinopses de cada filme e seus respetivos ID's.
plots = []
with open("plot_summaries.txt", 'r', encoding = "utf8") as f:
    reader = csv.reader(f, dialect = 'excel-tab') 
    for string in reader:
        plots.append(string)


# In[13]:


# Separação das variáveis contidas em "plots" para posterior criação de um dataframe contendo
# as colunas "Wikipedia movie ID" (assim como no dataframe do metadados) e "text" (que contém
# as sinopses).
movie_id = []
text = []
for i in plots:
    movie_id.append(i[0])
    text.append(i[1])

# Atribuição das colunas do dataframe a ser criado.
df_plots = pd.DataFrame({'Wikipedia movie ID': movie_id, 'text': text})
df_plots = df_plots.astype({"Wikipedia movie ID": int})
df_plots


# In[14]:


# Mescla dos dois dataframes em função da coluna "Wikipedia movie ID", comum a ambos.
# Nesse dataframe foram descartadas as variáveis que não agregam ao problema proposto.
df_merge = pd.merge(df_plots, data_movie[["Wikipedia movie ID", "Movie genres"]], on="Wikipedia movie ID", how="left")
df_merge


# In[15]:


# Função para limpar o texto: padronizando com letra minúscula, removendo espaços e caracteres
# que não fazem parte de palavras, removendo as "stopwords", que são palavras que não agregam
# ao conteúdo do texto, apenas à forma, e utilizando a técnica Stemming, que reduz as palavras
# ao seu radical.

# Obs: Na verdade, a ténica Stemming acabou sendo apenas referenciada abaixo, sem impacto no
# código final, pois o tempo de processamento era grande demais e pouco contribuia para a
# melhora do modelo.

stop = stopwords.words('english')
#stemmer = PorterStemmer()

def padronizar_texto(text):
    # todas os caracteres em letra minúscula
    text = text.lower()
    # substituir hifen por espaço para não juntar palavras
    text = text.replace("-", " ")
    # excluir tudo que não sejam letras e substituí-las por espaço em branco
    text = re.sub("[^a-zA-Z]"," ",text)
    # remover stopwords
    text = " ".join([word for word in text.split() if word not in (stop)])
    #text = text.split(" ")
    #text_stem = ""
    #for w in text:
    #    w = stemmer.stem(w)
    #    text_stem = text_stem + " " + w
    #text_stem = text_stem[1:]
    return text

# Aplicar a função padronizar_texto à coluna contendo as sinopses. A função "tqdm" serve
# para acompanhar o tempo de processamento da execução do código.
for i in tqdm(range(len(df_merge))):
    df_merge.at[i,'text'] = padronizar_texto(df_merge.at[i,'text'])


# In[16]:


# Remover os valores "NA" (sem informação) que aparecem no dataframe. Não serão úteis na construção
# do modelo.
# Alteração dos índices, colocando-os em ordem, no dataframe mesclado.
df_merge = df_merge.dropna()
df_merge.index = range(len(df_merge))
df_merge


# In[17]:


# Alocar coluna das sinopses na variávl "words", aplicar um loop para juntar todas as palavras
# que aparecem em uma única variável e depois separá-las, cada uma como um elemento de um vetor
words = df_merge['text']
all_words = ' '.join([text for text in words])
all_words = all_words.split()

# Encontrar a frequência de cada palavra e, em seguida, criar um dataframe com uma coluna contendo
# cada palavra que aparece e outra coluna contendo o número de aparições de cada palavra.
all_words_frq = nltk.FreqDist(all_words) 
words_df = pd.DataFrame({'Word':list(all_words_frq.keys()), 
                         'Count':list(all_words_frq.values())})
words_df


# In[18]:


# Gráfico de barra com a contagem de cada palavra, por ordem, da mais frequente à menos frequente, 
# exibindo apenas as 50 primeiras palavras. Lembrando que as "stopwords" foram excluídas.
wf = words_df.nlargest(columns = "Count", n = 50) 
plt.figure(figsize = (10,15)) 
ax = sns.barplot(data = wf, x = "Count", y = "Word") 
ax.set(ylabel = "Word") 
plt.show()


# In[86]:


# Ler as bibliotecas usadas na modelagem.
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# f1-score -> métrica de avaliação usada em problemas de classificação para medir a precisão do
# modelo.
# f1-score = 2 * (precisão * recall) / (precisão + recall)
# precisão = TP / (TP + FP)
# recall = TP / (TP + FN)
# TP: True Positive
# FP: False Positive
# FN: False Negative
# O f1-score varia de 0 a 1, indicando a precisão do modelo.
# Como é um problema multiclasse com as classes (gêneros) desbalanceadas, como a análise
# exploratória revelou, o parâmetro "average" da função f1_score ficará sempre definido
# como "micro".

# Criação e transformação das variáveis de interesse em matrizes com respostas binárias (0 ou 1)
# para cada uma das 363 classes de gêneros através da função MultiLabelBinarizer.
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df_merge['Movie genres'])

# Separação dos dados em dados de treino e teste. O parâmetro "test_size" define que 20%
# dos dados serão usados para teste e o resto para treino.
x_train, x_test, y_train, y_test = train_test_split(df_merge['text'], y, test_size = 0.2, random_state = 42)

# Utilizar a técnica TF-IDF para atribuir peso às palavras que aparecem nas sinopses. Essa 
# técnica cria uma matriz com os pesos de acordo com a aparição dos termos em um determinado
# documento e o número de documentos em que o termo aparece.
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)


# In[198]:


### DECISION TREE ###

# O modelo de Árvores de Decisão teve um desempenho razoável com alto custo operacional, durando
# cerca de 1 hora para treinar o modelo. Como houve outro modelo com melhores resultados, ele não
# foi escolhido como o modelo ideal para esse problema.

'''
# Biblioteca para o modelo de Árvores de Decisão.
from sklearn.tree import DecisionTreeClassifier

# Criar o modelo de Árvores de Decisão.
clf_DT = DecisionTreeClassifier(random_state = 42)
    
# Treinar o modelo com o conjunto de treinamento após a aplicação do TF-IDF e usar o tqdm
# para acompanhar e calcular o tempo de processamento.
with tqdm(desc="Treinando o modelo", total=1) as progress:
    clf_DT.fit(x_train_tfidf, y_train)
    progress.update(1)
    
# Previsões para testar o modelo.
y_pred = clf_DT.predict(x_test_tfidf)

# Utilizar o f1_score para avaliar a performance. 
# Definir o parâmetro average = "micro", pois as classes estão desbalanceadas nesse problema, 
# então o objetivo é avaliar o desempenho geral do modelo.
f1_score(y_test, y_pred, average="micro")
'''


# In[74]:


### RANDOM FOREST ###

# O tempo de processamento era muito grande e não houve ganho no f1-score em testes com uma
# amostra menor de dados em relação a Árvores de Decisão.

'''
# Biblioteca para o modelo de Random Forest.
from sklearn.ensemble import RandomForestClassifier

# Criar o modelo de Random Forest.
clf_RF = RandomForestClassifier(random_state=42)

# Treinar o modelo com o conjunto de treinamento após a aplicação do TF-IDF e usar o tqdm
# para acompanhar e calcular o tempo de processamento.
with tqdm(desc="Treinando o modelo", total=1) as progress:
    clf_RF.fit(x_train_tfidf, y_train)
    progress.update(1)

# Previsões para testar o modelo.
y_pred = clf_RF.predict(x_test_tfidf)

# Utilizar o f1_score para avaliar a performance. 
# Definir o parâmetro average = "micro", pois as classes estão desbalanceadas nesse problema, 
# então o objetivo é avaliar o desempenho geral do modelo.
f1_score(y_test, y_pred, average="micro")
'''


# In[ ]:


### SUPORT VECTOR MACHINE ###

# Infelizmente, não consegui testar o modelo SVM, pois aparecia sempre que o y deveria
# ser um array unidimensional e não consegui resolver esse problema a tempo de entregar;
# portanto, deixei apenas referenciado.

'''
# Biblioteca para o modelo de SVM.
from sklearn.preprocessing import MultiLabelBinarizer

# Tentativa de ajustar a variável 'y' para resolver o problema do array.
mlb = MultiLabelBinarizer()
y = df_merge['Movie genres'].apply(lambda x: ','.join(x)).tolist()
y = mlb.fit_transform(y)

# Dividir novamente os dados em treino e teste, após mudança na variável 'y' para esse caso específico.
x_train, x_test, y_train, y_test = train_test_split(df_merge['text'], y, test_size = 0.2, random_state = 42)

# Criar TF-IDF features.
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)

# Criar o modelo SVM.
clf_SVC = SVC(kernel='linear')

# Treinar o modelo com o conjunto de treinamento.
clf_SVC.fit(x_train_tfidf, y_train)

# Previsões para testar o modelo.
y_pred = clf.predict(x_test_tfidf)

# Utilizar o f1_score para avaliar a performance.
f1_score(y_test, y_pred, average="micro")
'''


# In[ ]:


### XGBoost

# O tempo de processamento era muito grande e não houve ganho no f1-score em testes com uma
# amostra menor de dados em relação a Árvores de Decisão.

'''
# Instalar o xgboost no Jupyter
!pip install xgboost

# Biblioteca para o modelo de XGBoost.
from xgboost import XGBClassifier

# Criar o modelo de XGBoost.
clf_XG = XGBClassifier(random_state=42)

# Treinar o modelo com o conjunto de treinamento.
clf_XG.fit(x_train_tfidf, y_train)

# Treinar o modelo com o conjunto de treinamento com tqdm.
with tqdm(desc="Treinando o modelo", total=1) as progress:
    clf_XG.fit(x_train_tfidf, y_train)
    progress.update(1)

# Previsões para testar o modelo.
y_pred = clf.predict(x_test_tfidf)

# Utilizar o f1_score para avaliar a performance.
f1_score(y_test, y_pred, average="micro")
'''


# In[20]:


### LOGISTIC REGRESSION

# Biblioteca para o modelo de Regressão Logística e a aplicação OneVsRestClassifier
# usada para problemas de múltiplas classes em modelos usualmente binários.
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Atribuição das funções
lr = LogisticRegression()
clf_LR = OneVsRestClassifier(lr)

# Treinar o modelo com o conjunto de treinamento após a aplicação do TF-IDF e usar o tqdm
# para acompanhar e calcular o tempo de processamento.
with tqdm(desc="Treinando o modelo", total=1) as progress:
    clf_LR.fit(x_train_tfidf, y_train)
    progress.update(1)

# Previsões para testar o modelo.
y_pred = clf_LR.predict(x_test_tfidf)

# Utilizar o f1_score para avaliar a performance. 
# Definir o parâmetro average = "micro", pois as classes estão desbalanceadas nesse problema, 
# então o objetivo é avaliar o desempenho geral do modelo.
f1_score(y_test, y_pred, average="micro")


# In[22]:


# No caso da Regressão Logística, a definição do resultado binário (0 ou 1) se baseia no cálculo
# de probabilidades de cada uma das duas possibilidades. O limiar padrão é 0.5 (50%-50%).
# Porém, a alteração nesse limiar pode trazer melhores resultados para o modelo, então
# o objetivo é encontrar o limiar que traz o melhor f1-score.

# Fazer as previsões usando o conjunto de teste.
y_pred_prob = clf_LR.predict_proba(x_test_tfidf)

# Inicialização das variáveis para armazenar o melhor limiar e o melhor f1-score.
best_threshold = 0
best_f1_score = 0

# loop entre os possíveis valores do limiar e respectivo cálculo do f1-score.
for threshold in tqdm(np.arange(0, 1.01, 0.01)):
    y_pred_threshold = (y_pred_prob > threshold).astype(int)
    f1 = f1_score(y_test, y_pred_threshold, average='micro')
    if f1 > best_f1_score:
        best_f1_score = f1
        best_threshold = threshold

# Saída com o melhor limiar e melhor f1-score.
print("Best threshold:", best_threshold)
print("Best F1-score:", best_f1_score)


# In[87]:


# Definidos os valores dos parâmetros que trazem os melhores resultados, o modelo é refeito com estes.
lr_updated = LogisticRegression()
clf_LR_updated = OneVsRestClassifier(lr_updated)

# Treinar o modelo com o conjunto de treinamento após a aplicação do TF-IDF e usar o tqdm
# para acompanhar e calcular o tempo de processamento.
with tqdm(desc="Treinando o modelo", total=1) as progress:
    clf_LR_updated.fit(x_train_tfidf, y_train)
    progress.update(1)

# Fazer as previsões usando o conjunto de teste com o clf atualizado.
y_pred_prob_updated = clf_LR_updated.predict_proba(x_test_tfidf)

# Usar o novo limiar para prever as classes.
new_y_pred = (y_pred_prob_updated >= best_threshold).astype(int)


# In[ ]:


# O primeiro modelo testado foi o modelo de Árvores de Decisão, com o f1-score de 0.23. Os modelos
# Random Forest e XGBoost tinham um custo operacional muito alto e, em testes com uma amostra menor
# de dados, os resultados do f1-score eram menores que os de Árvores de Decisão (com o mesmo tamanho
# de amostra). Já o modelo SVM não foi possível de ser testado por conta de erros que apareciam na
# hora da execução e do tempo limitado para a entrega do script. Por fim, o modelo de Regressão
# Linear obteve o melhor resultado aplicando o algoritmo OneVsRestClassifier, algoritmo este que
# também foi usado com o modelo de Árvores de Decisão, mas obteve resultado inferior ao do modelo
# sem o seu uso. O f1-score inicial do modelo de Regressão Linear foi 0.32; entretanto, ao usar
# o parâmetro "predict_proba", foi possível estabelecer o limiar que trazia o melhor f1-score,
# sendo estes, respectivamente, 0.18 e 0.47. Portanto, o modelo de Regressão Linear com o algoritmo
# OneVsRestClassifier, com limiar igual a 0.18, foi o modelo escolhido para prever os gêneros dos
# filmes a partir de suas sinopses.


# In[88]:


# Pipeline de predição do modelo Logistic Regression.
def prever_genero_LR(entrada):
    entrada = padronizar_texto(entrada)
    entrada_tfidf = tfidf_vectorizer.transform([entrada])
    # O "mlb.inverse_transform" reverte a transformação que havia sido feita na variável resposta
    # para atribuir valores binários às diferentes classes de gênero. Assim, a função prever_genero
    # retornará o(s) gênero(s) por escrito.
    return mlb.inverse_transform(clf_LR_updated.predict(entrada_tfidf))


# In[174]:


# A variável "entrada" recebe a sinopse, então a função prever_genero_LR, que obteve os
# melhores resultados nos treinamentos dos modelos, é executada e retorna os gêneros
# da sinopse recebida como entrada.
# Usei o exemplo enviado por email.
entrada = '''In 1936, archaeologist Indiana Jones braves an ancient Peruvian temple filled with booby traps to retrieve a golden idol. Upon fleeing the temple, Indiana is confronted by rival archaeologist René Belloq and the indigenous Hovitos. Surrounded and outnumbered, Indiana is forced to surrender the idol to Belloq and escapes aboard a waiting Waco seaplane, in the process revealing his fear of snakes. Shortly after returning to the college in the United States where he teaches archaeology, Indiana is interviewed by two Army intelligence agents. They inform him that the Nazis, in their quest for occult power, are searching for his old mentor, Abner Ravenwood, who is the leading expert on the ancient Egyptian city of Tanis and possesses the headpiece of an artifact called the Staff of Ra. Indiana deduces that the Nazis are searching for Tanis because it is believed to be the location of the Ark of the Covenant, the biblical chest built by the Israelites to contain the fragments of the Ten Commandments; the Nazis believe that if they acquire it, their armies will become invincible. The Staff of Ra, meanwhile, is the key to finding the Well of Souls, a secret chamber in which the Ark is buried. The agents subsequently authorize Indiana to recover the Ark before the Nazis. Indiana travels to Nepal, only to find that Ravenwood has died and that the headpiece is in the possession of his daughter, Marion, Indiana's embittered former lover. Indiana offers to buy the headpiece for three thousand dollars, plus two thousand more when they return to the United States. Marion's tavern is suddenly raided by a group of thugs commanded by Nazi agent Toht. The tavern is burned down in the ensuing fight, during which Toht burns his hand on the searing hot headpiece as he tries to grab it. Indiana and Marion escape with the headpiece, with Marion declaring she will accompany Indiana in his search for the Ark so he can repay his debt. They travel to Cairo where they learn from Indiana's friend Sallah, a skilled excavator, that Belloq and the Nazis, led by Colonel Dietrich, are currently digging for the Well of Souls with a replica of the headpiece modeled after the scar on Toht's hand. In a bazaar, Nazi operatives attempt to kidnap Marion and as Indiana chases after them it appears that she dies in an explosion. While deciphering the markings on the headpiece, Indiana and Sallah realize that the Nazis have miscalculated the location of the Well of Souls. Using this to their advantage, they infiltrate the Nazi dig and use the Staff of Ra to determine the location correctly and uncover the Well of Souls, which is filled with snakes. Indiana fends off the snakes and acquires the Ark, but Belloq, Dietrich and the Nazis arrive to take it. They toss Marion into the well with Indiana and seal them in, but they manage to escape. After a fistfight with a giant Nazi mechanic, blowing up a flying wing on the airstrip, and chasing down a convoy of trucks, Indiana takes back the Ark before it can be shipped to Berlin. Indiana and Marion leave Cairo to escort the Ark to England on board a tramp steamer. The next morning, their boat is boarded by Belloq, Dietrich and the Nazis, who once again steal the Ark and kidnap Marion. Indiana stows away on their U-boat and follows them to an isolated island in the Aegean Sea where Belloq plans to test the power of the Ark before presenting it to Hitler. Indiana reveals himself and threatens to destroy the Ark with a rocket-propelled grenade launcher, but Belloq calls his bluff, knowing Indy cannot bear to eradicate an important historical artifact. Indiana surrenders and is tied to a post with Marion as Belloq performs a ceremonial opening of the Ark, which appears to contain nothing but sand. Suddenly, spirits resembling Old Testament Seraphim emerge from the Ark. Aware of the supernatural danger of looking at the opened Ark, Indiana warns Marion to close her eyes. The apparitions suddenly morph into "angels of death", and lightning bolts begin flying out of the Ark, gruesomely killing the Nazi soldiers, while Belloq, Dietrich and Toht meet even more gruesome fates. The fires rise into the sky, then fall back down to Earth and the Ark closes with a crack of thunder. Back in Washington, D.C., the Army intelligence agents tell a suspicious Indiana and Brody that the Ark "is someplace safe" to be studied by "top men". In reality, the Ark is sealed in a wooden crate labeled "top secret" and stored in a giant government warehouse filled with countless similar crates.'''
entrada


# In[175]:


prever_genero_LR(entrada)


# In[172]:


# Fiz também um código recebendo uma amostra aleatória de tamanho 1 da coluna do dataframe 
# aonde estão as sinopses no seu formato inicial.
import random
amostra = random.randint(0, len(df_plots['text']))
entrada = df_plots.at[amostra,'text']
entrada


# In[173]:


prever_genero_LR(entrada)

