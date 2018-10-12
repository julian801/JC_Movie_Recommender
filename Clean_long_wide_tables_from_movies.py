
import pandas as pd

PATH = "movie_recommendations.xlsx"

df = pd.read_excel(PATH)

# produce clean unique index
df.sort_values(by=['Name', 'Genre', 'Reviewer'], inplace=True)
clean = df.drop_duplicates().dropna()

long = clean.set_index(['Name', 'Genre', 'Reviewer'])

# mean rating per genre
print(long.unstack(1).mean())
