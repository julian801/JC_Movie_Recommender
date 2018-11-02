
# coding: utf-8

# ## NMF = Not Monday night Football !

# In[1]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from random import randint
import numpy as np
from sklearn.decomposition import NMF
import pandas as pd
import random
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import csv


# # User Input

# In[2]:


#importing ratings and movies csv files
PATH2 = "ratings.csv"
PATH3 = "movies.csv"
ratings, movies_ind = pd.read_csv(PATH2), pd.read_csv(PATH3)


# In[3]:


# create an empty array the length of number of movies in system
user_ratings = np.zeros(9724)


# In[4]:


#format ratings dataframe
del ratings['timestamp']
ratings.set_index(['userId','movieId'], inplace=True)
ratings = ratings.unstack(0)


# In[5]:


ratings_count = ratings.count(axis=1) #count the number of ratings for each movie as a measure of popularity
top = pd.DataFrame(ratings_count.sort_values(ascending = False).head(10)) #create a dataframe of the top 20 most popular movies


# In[6]:


top.reset_index(inplace=True)


# In[7]:


movies_ind.set_index('movieId',inplace=True)


# In[8]:


top_movies_g = movies_ind.loc[top['movieId']]['title'].values


# ## Of the following movies, rate all that you have seen on a scale of 1-5. 
# ## If you have not seen a movie, rate 0.

# In[9]:


#creates a list of ratings for the prompted movies
user_input = []
for i in range(0,10):
    answer = int(input("How would you rate " + str(top_movies_g[i])))
    if answer > 5:
        answer = 5
    elif answer < 0:
        answer = 0
    user_input.append(answer)


# In[10]:


movies_ind.reset_index(inplace=True)


# In[11]:


top_movies_index = movies_ind.index[top['movieId']].values


# In[12]:


# inputs user rating into large array (9,000+ count) at appropriate indexes
for i in range(0,10):
    user_ratings[top_movies_index[i]] = user_input[i]


# # NMF Modeling

# In[13]:


ratings = ratings.fillna(0)
ratings = ratings["rating"]
ratings = ratings.transpose()


# In[14]:


ratings.head(2)


# In[15]:


R = pd.DataFrame(ratings)
# model assumes R ~ PQ'
model = NMF(n_components=5, init='random', random_state=10)
model.fit(R)

P = model.components_  # Movie feature
Q = model.transform(R)  # User features


# In[16]:


query = user_ratings.reshape(1,-1)


# In[17]:


t=model.transform(query)


# In[18]:


# prediction movie ratings of input user
outcome = np.dot(t,P)


# In[19]:


outcome=pd.DataFrame(outcome)


# In[20]:


outcome = outcome.transpose()


# In[21]:


outcome['movieId'] = movies_ind['movieId']


# In[22]:


outcome = outcome.rename(columns={0:'rating'})


# In[23]:


# top 100 ratings from predictions list
top = outcome.sort_values(by='rating',ascending=False).head(100)


# # Selecting a Movie

# In[24]:


# collects titles of the top movie predictions
top_movie_recs = movies_ind.loc[top['movieId']]['title'].values


# # Selecting a Movie with Genre Input

# In[25]:


#importing genres
PATHG = "movie_genres_years.csv"
movie_genres = pd.read_csv(PATHG)


# In[26]:


# creates list of genres
genres = movie_genres.columns.values[3:22]


# In[27]:


# dictionary with keys equal to genre
b,c = {}, {}
for x in genres:
    key = x
    value = ''
    b[key],c[key] = value, value


# In[28]:


# fills keys with list of movies that belong to respective genre
for x in genres:
    li = []
    for id in top['movieId']:
        if id in list(movie_genres.loc[movie_genres[x] == 1]['movieId']):
            li.append(movies_ind[movies_ind['movieId']==id]['title'].values)
    c[x] = li


# In[29]:


#fills keys with random choice in the list of films within a genre
for x in genres:
    if len(c[x])>0:
        b[x] = c[x][randint(0, len(c[x])-1)][0]
    else:
        b[x] = ""


# In[30]:


# add an option for not choosing a genre
genres_for_q = np.append(genres, 'none')


# In[31]:


from fuzzywuzzy import process


# In[32]:


genre_answer = process.extractOne(input("What genre of film would you like to watch?"),genres_for_q)


# In[33]:


#picks a top movie of the selected genre
for x in genres:
    if genre_answer[0] == x:
        if len(b[x]) == 0:
            print('No ' +x+ ' recommedations')
        else:
            print('We recommend ' + b[x])
            
# if they don't want a specific genre
if genre_answer[0] == 'none':
        Select = top_movie_recs[randint(0, 4)]
        print('We recommend ' + Select)

