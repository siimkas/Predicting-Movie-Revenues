#Polynomial regression

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder   
from sklearn.linear_model import LinearRegression    
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('movies_data.csv')



features=df.loc[:, ['adjusted_budget','popularity','runtime', 'vote_average']]
y=df['adjusted_revenue'].to_numpy()

regr = LinearRegression(fit_intercept=False)
scale = StandardScaler()
features_scaled = scale.fit_transform(features,y)
features_scaled = np.hstack((features_scaled, np.atleast_2d(LabelEncoder().fit_transform(df['original_language'])).T))
features['language_encoded'] = LabelEncoder().fit_transform(df['original_language'])


best_i_poly = 0
best_degree = 0
training_scores = []
validation_scores = []
best_validation_score = 0
corresponding_training_score = 0
test_error = 0
test_score = 0

for i in range(50):
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(features_scaled, y, test_size=0.2,shuffle=True, random_state=i)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2,shuffle=True, random_state=0)
    

    for degree_ in range(2,4):
        poly = PolynomialFeatures(degree=degree_)

        X_poly_train = poly.fit_transform(X_train)
        X_poly_val = poly.transform(X_val)
        X_poly_test = poly.transform(X_test)

        regr.fit(X_poly_train,y_train)

        y_pred_train = regr.predict(X_poly_train)
        y_pred_val = regr.predict(X_poly_val)
        

        training_score = regr.score(X_poly_train,y_train)
        training_scores.append(training_score)
        validation_score = regr.score(X_poly_val,y_val)
        validation_scores.append(validation_score)
        
        if(i == 0 and degree_==2):
            best_validation_score=validation_score

        if validation_score>best_validation_score: 
            
            best_i_poly = i
            best_validation_score = validation_score
            corresponding_training_score = training_score
            best_degree = degree_
            y_pred_test = regr.predict(X_poly_test)
            test_error = mean_absolute_error(y_test,y_pred_test)
            test_score = regr.score(X_poly_test, y_test)



median_training_score = np.median(training_scores)
median_validation_score = np.median(validation_score)

min_training_score = np.min(training_scores)
max_training_score = np.max(training_scores)
min_validation_score = np.min(validation_scores)
max_validation_score = np.max(validation_scores)

print('Results for Polynomial regression')
print('After 50 random datasets: \n \
    Minimum training score: '+str(round(min_training_score,2))+'\n \
    Maximum training score: '+ str(round(max_training_score,2))+ '\n \
    Median training score: '+ str(round(median_training_score,2))+ '\n\n \
    Minimum validation score: '+str(round(min_validation_score,2))+'\n \
    Maximum validation score: '+ str(round(max_validation_score,2))+ '\n \
    Median validation score: '+ str(round(median_validation_score,2))+ '\n\n \
    Final model (best validation score):'+'\n \
    Degree: '+str(best_degree)+'\n \
    Training score: '+str(round(corresponding_training_score,2))+ '\n \
    Validation score: '+str(round(best_validation_score,2))+'\n \
    Test score: '+str(round(test_score,2))+'\n \
    Test error: '+str(round(test_error,2))+'$\n \
    Mean inflation adjusted revenue: '+str(round(np.mean(y),2))+'$')

