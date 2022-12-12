#Script for feature plotting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder   
from sklearn.feature_selection import r_regression
from sklearn.linear_model import LinearRegression    
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split



df = pd.read_csv('movies_data.csv')

df['language_encoded'] = LabelEncoder().fit_transform(df['original_language'])

features=df.loc[:, ['adjusted_budget','popularity','runtime', 'vote_average', 'language_encoded']]
y=df['adjusted_revenue'].to_numpy()
coefficients = r_regression(features,y)

fig, axes = plt.subplots(2, 2, figsize=(14,5)) 
fig.tight_layout(pad=5.0)
axes[0][0].scatter(df['adjusted_budget'],df['adjusted_revenue']) 
axes[0][0].set_xlabel("Inflation adjusted budget",size=15)
axes[0][0].set_ylabel("Adjusted revenue",size=15)
axes[0][0].set_title("Budget vs revenue\nPearson correlation coefficient: "+str(round(coefficients[0],2)),size=15)

axes[0][1].scatter(df['popularity'],df['adjusted_revenue']) 
axes[0][1].set_xlabel("Popularity",size=15)
axes[0][1].set_ylabel("Adjusted revenue",size=15)
axes[0][1].set_title("Popularity vs revenue\nPearson correlation coefficient: "+str(round(coefficients[1],2)),size=15)

axes[1][0].scatter(df['runtime'],df['adjusted_revenue']) 
axes[1][0].set_xlabel("Runtime",size=15)
axes[1][0].set_ylabel("Adjusted revenue",size=15)
axes[1][0].set_title("Runtime vs revenue\nPearson correlation coefficient: "+str(round(coefficients[2],2)),size=15)


axes[1][1].scatter(df['vote_average'],df['adjusted_revenue']) 
axes[1][1].set_xlabel("Rating",size=15)
axes[1][1].set_ylabel("Adjusted revenue",size=15)
axes[1][1].set_title("Rating vs revenue\nPearson correlation coefficient: "+str(round(coefficients[3],2)),size=15)

plt.show()

languages = df['original_language'].unique()

language_revenues = {}

for language in languages:
    data = df[df['original_language']==language]
    mean = data['adjusted_revenue'].to_numpy().mean()
    language_revenues[mean] = language

language_revenues = dict(sorted(language_revenues.items(),reverse=True))
plt.bar(language_revenues.values(),language_revenues.keys())
plt.xlabel('Language')
plt.ylabel('Mean revenue')
plt.title('Mean revenue by language')

plt.show()

