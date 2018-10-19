# write simple ui
# print explanatory text
# ask user for genre
# and fav reviewer
# either can be empty
# return the best scoring hit

import pandas as pd

PATH = "movie_recommendations.xlsx"

df = pd.read_excel(PATH)

# produce clean unique index
df.sort_values(by=['Name', 'Genre', 'Reviewer'], inplace=True)
clean = df.drop_duplicates().dropna()

long = clean.set_index(['Name', 'Genre', 'Reviewer'])

# mean rating per genre
print(long.unstack(1).mean())

print("""

    A-Team recommender: Let BA find you a movie, fool!
    
    Enter your favourite genre of reviewier:

""")

print("enter a genre (or press enter to skip): ", end="")
genre = imput()

print("enter a reviewer (or press enter to skip): ", end="")
reviewer = imput()

if genre: #boolean evaluating whether the genre is null or not
    gdf = df[df['genre']==genre]
else:
    g = df

result = g.sort_values('Rating', ascending=False, inplace=True)
print(g.head(3).sample(1))

