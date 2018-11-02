
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
    user_input.append(answer)


# In[10]:


movies_ind.reset_index(inplace=True)


# In[11]:


top_movies_index = movies_ind.index[top['movieId']].values


# In[12]:


top_movies_index


# In[13]:


# inputs user rating into large array (9,000+ count) at appropriate indexes
for i in range(0,10):
    user_ratings[top_movies_index[i]] = user_input[i]


# # NMF Modeling

# In[14]:


PATH4 = "movies.csv"
movies_ind = pd.read_csv(PATH4)

ratings = ratings.fillna(0)
ratings = ratings["rating"]
ratings = ratings.transpose()


# In[15]:


ratings.head(2)


# In[16]:


R = pd.DataFrame(ratings)
# model assumes R ~ PQ'
model = NMF(n_components=5, init='random', random_state=10)
model.fit(R)

P = model.components_  # Movie feature
Q = model.transform(R)  # User features


# In[17]:


query = user_ratings.reshape(1,-1)


# In[18]:


t=model.transform(query)


# In[19]:


outcome = np.dot(t,P)


# In[20]:


outcome=pd.DataFrame(outcome)


# In[21]:


outcome = outcome.transpose()


# In[22]:


outcome['movieId'] = movies_ind['movieId']


# In[23]:


outcome = outcome.rename(columns={0:'rating'})


# In[24]:


outcome


# In[60]:


top = outcome.sort_values(by='rating',ascending=False).head(100)


# # Selecting a Movie with no Genre Input

# In[61]:


top_movie_recs = movies_ind.loc[top['movieId']]['title'].values


# In[62]:


Select = top_movie_recs[randint(0, 4)]
Select


# # Selecting a Movie with Genre Input

# In[63]:


#importing genres
PATHG = "movie_genres_years.csv"
movie_genres = pd.read_csv(PATHG)


# In[64]:


# list of all movie Ids belonging to certain genres
adventure_movies = list(movie_genres.loc[movie_genres['Genre_Adventure'] == 1]['movieId'])
animation_movies= list(movie_genres.loc[movie_genres['Genre_Animation'] == 1]['movieId'])
children_movies= list(movie_genres.loc[movie_genres['Genre_Children'] == 1]['movieId'])
comedy_movies= list(movie_genres.loc[movie_genres['Genre_Comedy'] == 1]['movieId'])
fantasy_movies= list(movie_genres.loc[movie_genres['Genre_Fantasy'] == 1]['movieId'])
romance_movies= list(movie_genres.loc[movie_genres['Genre_Romance'] == 1]['movieId'])
drama_movies= list(movie_genres.loc[movie_genres['Genre_Drama'] == 1]['movieId'])
action_movies= list(movie_genres.loc[movie_genres['Genre_Action'] == 1]['movieId'])
crime_movies= list(movie_genres.loc[movie_genres['Genre_Crime'] == 1]['movieId'])
thriller_movies= list(movie_genres.loc[movie_genres['Genre_Thriller'] == 1]['movieId'])
horror_movies= list(movie_genres.loc[movie_genres['Genre_Horror'] == 1]['movieId'])
mystery_movies= list(movie_genres.loc[movie_genres['Genre_Mystery'] == 1]['movieId'])
scifi_movies= list(movie_genres.loc[movie_genres['Genre_Sci-Fi'] == 1]['movieId'])
war_movies= list(movie_genres.loc[movie_genres['Genre_War'] == 1]['movieId'])
musical_movies= list(movie_genres.loc[movie_genres['Genre_Musical'] == 1]['movieId'])
documentary_movies= list(movie_genres.loc[movie_genres['Genre_Documentary'] == 1]['movieId'])
imax_movies= list(movie_genres.loc[movie_genres['Genre_IMAX'] == 1]['movieId'])
western_movies= list(movie_genres.loc[movie_genres['Genre_Western'] == 1]['movieId'])
noir_movies= list(movie_genres.loc[movie_genres['Genre_Film-Noir'] == 1]['movieId'])


# In[65]:


genres = movie_genres.columns.values[3:22]


# In[66]:


a = {}
for x in genres:
    key = x
    value = ''
    a[key] = value 


# In[67]:


ad = []
an = []
ch = []
co = []
fa = []
ro = []
dr = []
ac = []
cr = []
th = []
ho = []
my = []
sc = []
wa = []
mu = []
do = []
im = []
we = []
fi = []

for x in top['movieId']:
    if x in adventure_movies:
        ad.append(movies_ind[movies_ind['movieId']==x]['title'].values)
    a['Genre_Adventure'] = ad
for x in top['movieId']:
    if x in animation_movies:
        an.append(movies_ind[movies_ind['movieId']==x]['title'].values)
    a['Genre_Animation'] = an
for x in top['movieId']:
    if x in children_movies:
        ch.append(movies_ind[movies_ind['movieId']==x]['title'].values)
    a['Genre_Children'] = ch
for x in top['movieId']:
    if x in comedy_movies:
        co.append(movies_ind[movies_ind['movieId']==x]['title'].values)
    a['Genre_Comedy'] = co
for e in top['movieId']:
    if e in fantasy_movies:
        fa.append(movies_ind[movies_ind['movieId']==e]['title'].values)
    a['Genre_Fantasy'] = fa
for f in top['movieId']:
    if f in romance_movies:
        ro.append(movies_ind[movies_ind['movieId']==f]['title'].values)
    a['Genre_Romance'] = ro
for g in top['movieId']:
    if g in drama_movies:
        dr.append(movies_ind[movies_ind['movieId']==g]['title'].values)
    a['Genre_Drama'] = dr
for h in top['movieId']:
    if h in action_movies:
        ac.append(movies_ind[movies_ind['movieId']==h]['title'].values)
    a['Genre_Action'] = ac
for i in top['movieId']:
    if i in crime_movies:
        cr.append(movies_ind[movies_ind['movieId']==i]['title'].values)
    a['Genre_Crime'] = cr
for j in top['movieId']:
    if j in thriller_movies:
        th.append(movies_ind[movies_ind['movieId']==j]['title'].values)
    a['Genre_Thriller'] = th
for k in top['movieId']:
    if k in horror_movies:
        ho.append(movies_ind[movies_ind['movieId']==k]['title'].values)
    a['Genre_Horror'] = ho
for l in top['movieId']:
    if l in mystery_movies:
        my.append(movies_ind[movies_ind['movieId']==l]['title'].values)
    a['Genre_Mystery'] = my
for m in top['movieId']:
    if m in scifi_movies:
        sc.append(movies_ind[movies_ind['movieId']==m]['title'].values)
    a['Genre_Sci-Fi'] = sc
for n in top['movieId']:
    if n in war_movies:
        wa.append(movies_ind[movies_ind['movieId']==n]['title'].values)
    a['Genre_War'] = wa
for o in top['movieId']:
    if o in musical_movies:
        mu.append(movies_ind[movies_ind['movieId']==o]['title'].values)
    a['Genre_Musical'] = mu
for p in top['movieId']:
    if p in documentary_movies:
        do.append(movies_ind[movies_ind['movieId']==p]['title'].values)
    a['Genre_Documentary'] = do
for q in top['movieId']:
    if q in imax_movies:
        im.append(movies_ind[movies_ind['movieId']==q]['title'].values)
    a['Genre_IMAX'] = im
for r in top['movieId']:
    if r in western_movies:
        we.append(movies_ind[movies_ind['movieId']==r]['title'].values)
    a['Genre_Western'] = we
for s in top['movieId']:
    if s in noir_movies:
        fi.append(movies_ind[movies_ind['movieId']==s]['title'].values)
    a['Genre_Film-Noir'] = fi
        


# In[76]:


adventure_rec = a['Genre_Adventure'][randint(0, len(a['Genre_Adventure'])-1)][0]
animation_rec = a['Genre_Animation'][randint(0, len(a['Genre_Animation'])-1)][0]
children_rec = a['Genre_Children'][randint(0, len(a['Genre_Children'])-1)][0]
comedy_rec = a['Genre_Comedy'][randint(0, len(a['Genre_Comedy'])-1)][0]
fantasy_rec = a['Genre_Fantasy'][randint(0, len(a['Genre_Fantasy'])-1)][0]
romance_rec = a['Genre_Romance'][randint(0, len(a['Genre_Romance'])-1)][0]
drama_rec = a['Genre_Drama'][randint(0, len(a['Genre_Drama'])-1)][0]
action_rec = a['Genre_Action'][randint(0, len(a['Genre_Action'])-1)][0]
crime_rec = a['Genre_Crime'][randint(0, len(a['Genre_Crime'])-1)][0]
thriller_rec = a['Genre_Thriller'][randint(0, len(a['Genre_Thriller'])-1)][0]
horror_rec = a['Genre_Horror'][randint(0, len(a['Genre_Horror'])-1)][0]
mystery_rec = a['Genre_Mystery'][randint(0, len(a['Genre_Mystery'])-1)][0]
scifi_rec = a['Genre_Sci-Fi'][randint(0, len(a['Genre_Sci-Fi'])-1)][0]
war_rec = a['Genre_War'][randint(0, len(a['Genre_War'])-1)]
musical_rec = a['Genre_Musical'][randint(0, len(a['Genre_Musical'])-1)][0]
imax_rec = a['Genre_IMAX'][randint(0, len(a['Genre_IMAX'])-1)][0]
western_rec = a['Genre_Western'][randint(0, len(a['Genre_Western'])-1)][0]
noir_rec = a['Genre_Film-Noir'][randint(0, len(a['Genre_Film-Noir'])-1)][0]


# In[69]:


from fuzzywuzzy import process


# In[83]:


genre_answer = process.extractOne(input("What genre of film would you like to watch?"),genres)


# In[84]:


if genre_answer[0] == 'Genre_Adventure':
    if len(a['Genre_Adventure']) == 0:
        print('No adventure recommendations')
    else:
        print('We recommend ' + adventure_rec)
if genre_answer[0] == 'Genre_Animation':
    if len(a['Genre_Animation']) == 0:
        print('No animation recommendations')
    else:
        print('We recommend ' + animation_rec)
if genre_answer[0] == 'Genre_Children':
    if len(a['Genre_Children']) == 0:
        print('No chilren\'s movies recommendations')
    else:
        print('We recommend ' + children_rec)
if genre_answer[0] == 'Genre_Comedy':
    if len(a['Genre_Comedy']) == 0:
        print('No comedy recommendations')
    else:
        print('We recommend ' + comedy_rec)
if genre_answer[0] == 'Genre_Fantasy':
    if len(a['Genre_Fantasy']) == 0:
        print('No fantasty recommendations')
    else:
        print('We recommend ' + fantasy_rec)
if genre_answer[0] == 'Genre_Romance':
    if len(a['Genre_Romance']) == 0:
        print('No romance recommendations')
    else:
        print('We recommend ' + romance_rec)
if genre_answer[0] == 'Genre_Drama':
    if len(a['Genre_Drama']) == 0:
        print('No drama recommendations')
    else:
        print('We recommend ' + drama_rec)
if genre_answer[0] == 'Genre_Action':
    if len(a['Genre_Action']) == 0:
        print('No action recommendations')
    else:
        print('We recommend ' + action_rec)
if genre_answer[0] == 'Genre_Crime':
    if len(a['Genre_Crime']) == 0:
        print('No crime recommendations')
    else:
        print('We recommend ' + crime_rec)
if genre_answer[0] == 'Genre_Thriller':
    if len(a['Genre_Thriller']) == 0:
        print('No thriller recommendations')
    else:
        print('We recommend ' + thriller_rec)
if genre_answer[0] == 'Genre_Horror':
    if len(a['Genre_Horror']) == 0:
        print('No horror recommendations')
    else:
        print('We recommend ' + horror_rec)
if genre_answer[0] == 'Genre_Mystery':
    if len(a['Genre_Mystery']) == 0:
        print('No mystery recommendations')
    else:
        print('We recommend ' + mystery_rec)
if genre_answer[0] == 'Genre_Sci-Fi':
    if len(a['Genre_Sci-Fi']) == 0:
        print('No sci-fi recommendations')
    else:
        print('We recommend ' + scifi_rec)
if genre_answer[0] == 'Genre_War':
    if len(a['Genre_War']) == 0:
        print('No war recommendations')
    else:
        print('We recommend ' + war_rec)
if genre_answer[0] == 'Genre_Musical':
    if len(a['Genre_Musical']) == 0:
        print('No musical recommendations')
    else:
        print('We recommend ' + musical_rec)
if genre_answer[0] == 'Genre_Western':
    if len(a['Genre_Western']) == 0:
        print('No western recommendations')
    else:
        print('We recommend ' + western_rec)
if genre_answer[0] == 'Genre_Film-Noir':
    if len(a['Genre_Film-Noir']) == 0:
        print('No film-noir recommendations')
    else:
        print('We recommend ' + noir_rec)

