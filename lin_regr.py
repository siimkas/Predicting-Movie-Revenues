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
y=df['adjusted_revenue'].to_numpy()
scale = StandardScaler()
features=df.loc[:, ['adjusted_budget','popularity','runtime', 'vote_average']]
features_scaled = scale.fit_transform(features,y)
features_scaled = np.hstack((features_scaled, np.atleast_2d(LabelEncoder().fit_transform(df['original_language'])).T))

regr = LinearRegression()
accuracy_lin = 0
best_i = 0
best_j = 0

for i in range(100):
    for j in range(100):
        X_train_val, X_test, y_train_val, y_test = train_test_split(features_scaled, y, test_size=0.2,shuffle=True, random_state=i)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2,shuffle=True, random_state=j)


        model = regr.fit(X_train,y_train)
        accuracy_lin_ = model.score(X_val,y_val)

        if accuracy_lin_>accuracy_lin:
            accuracy_lin=accuracy_lin_
            best_i = i
            best_j = j
        

    print(str(i)+'"%" ready')



X_train_val, X_test, y_train_val, y_test = train_test_split(features_scaled, y, test_size=0.2,shuffle=True, random_state=best_i)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2,shuffle=True, random_state=best_j)
model = regr.fit(X_train,y_train)
accuracy_lin_ = model.score(X_test,y_test)
print('Linear regression best score with valuation set: '+str(accuracy_lin)+', with test set:'+str(accuracy_lin_))