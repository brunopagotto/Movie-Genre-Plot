#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
data = pd.read_csv('character.metadata.tsv',sep='\t',header=None)
data


# In[90]:


import pandas as pd
data_movie = pd.read_csv('movie.metadata.tsv',sep='\t',header=None)
data_movie.columns = ['Wikipedia movie ID','Freebase movie ID','Movie name','Movie release date',
                     'Movie box office revenue','Movie runtime','Movie languages','Movie countries',
                     'Movie genres']
data_movie


# In[35]:


data_movie.describe


# In[95]:


list_genres = data_movie.at[0,'Movie genres']
list_genres


# In[97]:


list_genres = data_movie.at[0,'Movie genres'][1:-2]
list_genres


# In[82]:


list_genres = data_movie.at[0,'Movie genres'].replace('": "','", "').split('", "')
list_genres


# In[83]:


list_genres = list_genres[1::2]
list_genres


# In[88]:


list_genres[-1]


# In[85]:


colunas = ['Movie languages','Movie countries','Movie genres']

def reduzir_string(col_name):
    for i in range(len(data_movie)):
        list_genres = data_movie.at[i,col_name].replace('": "','", "').split('", "')
        list_genres = list_genres[1::2]
        data_movie.at[i,col_name] = list_genres
        
for col in colunas:
        reduzir_string(col)


# In[86]:


data_movie


# In[ ]:


data_movie.at[0,'Movie genres'][0]


# In[28]:


data_movie[data_movie['Wikipedia movie ID'] == 54166]


# In[16]:


fileObject = open("plot_summaries.txt", "r", encoding="utf8")
sinopses = fileObject.read()
print(sinopses)


# In[22]:


f=open("plot_summaries.txt", "r", encoding="utf8")
lines=f.readlines()
result=[]
for x in lines:
    result.append(x.split(' ')[1])
f.close()


# In[13]:


with open("plot_summaries.txt", 'r', encoding="utf8") as fp:
    x = len(fp.readlines())
    print('Total lines:', x) # 8
    
with open("plot_summaries.txt", 'r', encoding="utf8") as fp:
    num_lines = sum(1 for line in fp if line.rstrip())
    print('Total lines:', num_lines)  # 8


# In[19]:


myfile = open("plot_summaries.txt", 'r', encoding="utf8")
myline = myfile.readline()
print(myline)
myfile.close()


# In[21]:


myfile = open("plot_summaries.txt", 'r', encoding="utf8")
myline1 = myfile.readline()
myline2 = myfile.readline()
print(myline1)
print(myline2)
myfile.close()


# In[24]:


token = open("plot_summaries.txt", 'r', encoding="utf8")
linestoken=token.readlines()
tokens_column_number = 0
resulttoken=[]
for x in linestoken:
    resulttoken.append(x.split()[tokens_column_number])
token.close()
print(resulttoken)


# In[ ]:


ab = []
for i in tqdm(range(len(data_movie))):
    ab = set(itertools.chain(list(ab),genres[i]))
    
list(ab)


# In[ ]:


import itertools
ab = itertools.chain(genres[0],genres[1])
ab = itertools.chain(list(ab),genres[2])
list(ab)


# In[ ]:


words = ["want", "wanted", "wants", "wanting"]
stemmer = PorterStemmer()
for word in words:
    print(word + " = " + stemmer.stem(word))


# In[ ]:


data_movie_dummies = pd.get_dummies(list(genres_unique))     # Convert column to dummies
print(data_movie_dummies)                           # Print dummies


# In[ ]:


data_all = pd.concat([data, data_dummies],    # Combine original data & dummies
                     axis = 1)
print(data_all)                               # Print all data


# In[ ]:


y = np.array(df_merge['Movie genres'])
y = y.reshape(-1, len(mlb.classes_))

