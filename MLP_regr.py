#MLP regression

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('movies_data.csv')

df['language_encoded'] = LabelEncoder().fit_transform(df['original_language'])

features=df.loc[:, ['adjusted_budget','popularity','runtime', 'vote_average', 'language_encoded']]
y=df['adjusted_revenue'].to_numpy()

num_layers = [1,2,3,4]   
num_neurons = [3,4,5,6,7,8,9,10]           

best_i_mlp = 0
training_scores = []
validation_scores = []
best_validation_score = 0
corresponding_training_score = 0
test_error = 0
best_nr_layers = 0
best_nr_neurons = 0
test_score = 0


for j in range(50):
    X_train_val, X_test, y_train_val, y_test = train_test_split(features, y, test_size=0.2,shuffle=True, random_state=j)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2,shuffle=True, random_state=0)
    for  i,layers in enumerate(num_layers):
        for neurons in num_neurons:

            hidden_layer_sizes = tuple([neurons]*layers) 
            
            mlp_regr = MLPRegressor(hidden_layer_sizes,max_iter = 2000,random_state=0)
            mlp_regr.fit(X_train,y_train)
            
            training_score = mlp_regr.score(X_train,y_train)
            validation_score = mlp_regr.score(X_val,y_val)
            
            training_scores.append(training_score)
            validation_scores.append(validation_score)

            if (validation_score>best_validation_score):
                best_validation_score = validation_score
                best_validation_score = validation_score
                corresponding_training_score = training_score
                best_nr_layers = layers
                best_nr_neurons = neurons
                y_pred_test = mlp_regr.predict(X_test)
                test_error = mean_absolute_error(y_test,y_pred_test)
                test_score = mlp_regr.score(X_test,y_test)



mean_training_score = np.median(training_scores)
mean_validation_score = np.median(validation_score)

min_training_score = np.min(training_scores)
max_training_score = np.max(training_scores)
min_validation_score = np.min(validation_scores)
max_validation_score = np.max(validation_scores)


print('Results for MLP regression: \n \
After 50 random datasets: \n \
    Minimum training score: '+str(round(min_training_score,2))+'\n \
    Maximum training score: '+ str(round(max_training_score,2))+ '\n \
    Median training score: '+ str(round(mean_training_score,2))+ '\n\n \
    Minimum validation score: '+str(round(min_validation_score,2))+'\n \
    Maximum validation score: '+ str(round(max_validation_score,2))+ '\n \
    Median validation score: '+ str(round(mean_validation_score,2))+ '\n\n \
    Final model (best validation score):'+'\n \
    Number of layers: '+str(best_nr_layers)+'\n \
    Number of neurons: '+str(best_nr_neurons)+'\n \
    Training score: '+str(round(corresponding_training_score,2))+ '\n \
    Validation score: '+str(round(best_validation_score,2))+'\n \
    Test score: '+str(round(test_score,2))+'\n \
    Test error: '+str(round(test_error,2))+'$\n \
    Mean inflation adjusted revenue: '+str(round(np.mean(y),2))+'$')