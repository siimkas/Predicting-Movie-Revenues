# Predicting Movie Revenues
## Introduction
The movie and production industry in America is a multibillion dollar industry. When it comes to predicting the revenue of a movie, a variety of different factors play a role in determining whether a movie is a “box office hit”. Some of the factors include the language of the movie, the production budget, rating of the movie etc. An interesting project idea we thought would be to analyse the extent to which some of the aforementioned factors can predict the total revenue of a movie throughout its screening, through implementing a supervised machine learning model.
Section 2 (“Problem Formulation”) discusses the specifics of the problem, information about the feature, data set and where the data set comes from. Section 3 (“Methods”) discusses the methods used to examine the problem. Section 4 (“Results”) and 5 (“Conclusion”) discuss the results and the conclusions that can be drawn from the findings, and Section 6 (“References”) contains the references.


## 2. Problem Formulation
The aim is to train a supervised machine learning model that will predict the total revenue (in US dollars) a movie will earn throughout its screenings, which will be the label. To gather high-quality data, the Python Requests [1] library was used to fetch 4076 data points from The Movie Database (TMDB) [2] using their API [3]. The features collected for each data point are the popularity index (continuous) [4], budget (continuous), runtime (continuous), average rating (continuous, ranging from 0 to 10), and the original language(categorical).

### 2.1 The dataset
To train and test this model accurately, the data was chosen from the 1st of January 2000 to the 31st of December 2019. This was done to have data that has matured enough (movie revenues take months or even a year to accumulate) and because of the mass revenue losses in the movie industry due to the COVID-19 lockdowns. Each data point in this set corresponds to one movie title. To achieve better accuracy and validity, only movies with at least 100 votes were considered in order to give the popularity index some credibility. Furthermore, the data set was filtered so that movies with no budget and revenue data were not selected. This filtering process resulted in 4076 data points.

## 3. Methods

### 3.1 Feature selection -and engineering
The features were selected by using the Pearson Correlation Coefficient [5] and data visualisation. The graphs and values that the decisions were based on are the following:

![image](https://user-images.githubusercontent.com/95539000/208319921-05752beb-9acb-4ae9-ac7d-17ab0a6819b9.png)
Figure 1: Original language and mean revenue
![image](https://user-images.githubusercontent.com/95539000/208319957-0cd5f3c8-17d2-4fda-8d82-73658cc46dc5.png)
Figure 2: Correlation of budget, popularity, runtime, and rating regards to revenue


All of the features represented in Figure 1 show both a positive correlation coefficient (included in the graphs) regards to revenue as well as a recognizable pattern, therefore the selection of these features is justified. In addition, a clear pattern can be seen in Figure 2, where the revenues are compared to their original language.
Before the features can be used and to increase the accuracy of the model they need to be modified. To transform languages to numerical values the Python scikit-learn library’s LabelEncoder [6] was used. In addition, budget and revenue values were adjusted to inflation using the yearly Consumer Price Index (CPI) released by the US Federal Reserve [7] in order to make the monetary values comparable throughout the 20-year time interval; this was done with the help of the cpi library [8]. Finally, the whole feature set is standardised using StandardScaler [9].

### 3.2 Methods, Resulting models and Loss functions
The first method used in this project is Polynomial regression [10] because of the non-linear relationship between the features and the label clearly seen in Figures 1 and 2. The degree of the Polynomial regression should not be more than 3 as it will result in overfitting the model. A polynomial has n-1 extremums, where n is the degree of the polynomial, and therefore the data visualisation doesn’t justify using a degree more than 3, as the number of extremums is clearly less than 4.
The second method utilised is the Multi-layer Perceptron (MLP) regression [11]. The reason being that it is also used for regression prediction problems, is widely used in the industry and has been proven to be a reliable method. Therefore, it was decided to use it to compare the difference in accuracy compared to the polynomial regression model.
With regards to the loss functions, the Polynomial regression uses the residual sum of squares error function [12] and MLP regression uses the squared error function [13]. Both have been chosen because the libraries used for this project apply them as the default option. As for computing the training-, validation-, and test errors, it was decided to use the regression libraries’ score function, which returns the coefficient of determination ranging from -1 to 1 (1 being the best score). This was done to make the interpretation of the results easier since the revenues can reach to billions of dollars which would make the squared errors immensely large. Finally, to make the results easily understandable to readers, a mean absolute error [14] is calculated in dollars for the test set.

### 3.3 Model validation
To validate the model, the data has been split into three distinct sets: training-, validation -and test set. The training set is used to train the model that will be then validated with the separate validation set. Usually, the final testing is done with the most recent data, but because movie revenue data takes a long time to mature and recent movie revenues have been affected by lockdowns all over the world, a randomised set must be used instead. The training and validation sets are also randomised from the 20-year period to find a uniform trend. Since there are a reasonable amount of data points, the test set is chosen to be 20 % of the whole set, which is a common percentage used in the industry [15]. The leftover 80% is then also split with the same ratio, 80% being the training set and 20% being the validation set.
To find the best model, it will be trained with multiple rounds of randomised data sets. Each round in turn consists of multiple rounds where some characteristics are adjusted for both methods. For Polynomial regression it is the degree and for MLP regression it is the number of hidden layers and number of neurons in each layer. At the end of each round, the training- and validation scores along with the arguments are recorded before moving on to the next round. Finally, the model with the best validation score is chosen and a final test is performed on the test set, which has not influenced the process in any way. Only then there is confidence in the final accuracy of the model and that it can be used for predicting future data. The training score does not influence the choice of the model to avoid overfitting the data during the process. Due to time constraints, the process considers only 50 different randomised sets of data.

## 4. Results

![image](https://user-images.githubusercontent.com/95539000/208320240-d3442214-3dfa-4a90-a799-65fd9586d0c6.png)

Figure 3: Polynomial model results

![image](https://user-images.githubusercontent.com/95539000/208320272-de8c066c-044e-4e70-a140-2c350a614c2f.png)

Figure 4: MLP model results
  
From the obtained results, which can be seen in Figure 3 and 4, it is clear that Polynomial regression is performing slightly better with the current set of features. The final test error for the Polynomial regression is ~67 mil $ while it is ~95,4 mil $ for MLP regression.
The training scores do not give significant meaning to the results as they are not used to choose the final model. However, the MLP regression minimum training score and both minimum validation scores indicate that there might be a problem with outliers in the data set as the difference between the median scores is considerably large. It might also indicate that more feature engineering is needed.

## 5. Conclusion
Based on this project, which has been done with limited knowledge, predicting movie revenues is a worthy problem for machine learning since it has been now shown that somewhat meaningful results can be retrieved with even primitive knowledge in the machine learning space. The final training error and training score clearly hint that the final model is not optimal and there is room for improvement.
For future improvements, loss functions (both for hypothesis space and calculating errors) that take into account the outliers should be considered, for reasons mentioned in the previous section. In addition, more thorough feature engineering could improve the results of both models in addition to increasing the number of randomised sets used in the process. And finally, gathering better quality data from a more complete database would most likely be an advantage, for example data from imdb.com [17].

## 6. References
[1] R. Python, ‘Python’s Requests Library (Guide) – Real Python’. [Online]. Available: https://realpython.com/python-requests/ (accessed Oct. 09, 2022).  
[2] ‘The Movie Database (TMDB)’.[Online]. Available: https://www.themoviedb.org/ (accessed Oct. 09, 2022).  
[3] ‘The Movie Database API’. Accessed: Sep. 29, 2022. [Online]. Available: https://developers.themoviedb.org/3/getting-started/introduction  
[4] ‘The Movie Database API’, API Docs. [Online]. Available: https://developers.themoviedb.org/3/getting-started/popularity (accessed Oct. 09, 2022).  
[5] ‘sklearn.feature_selection.r_regression’, scikit-learn. [Online]. Available: https://scikit-learn/stable/modules/generated/sklearn.feature_selection.r_regression.html (accessed Oct. 09, 2022).  
[6] ‘sklearn.preprocessing.LabelEncoder’, scikit-learn. [Online]. Available: https://scikit-learn/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html (accessed Oct. 09, 2022).  
[7] H. J. Ahn and C. Fulton, ‘Index of Common Inflation Expectations’, Feb. 2020, Accessed: Oct. 09, 2022. [Online]. Available: https://www.federalreserve.gov/econres/notes/feds-notes/index-of-common-inflation-expectatio ns-20200902.html  
[8] ‘cpi — cpi documentation’. [Online]. Available: https://palewi.re/docs/cpi/ (accessed Oct. 09, 2022).  
[9] ‘sklearn.preprocessing.StandardScaler’, scikit-learn. [Online]. Available: https://scikit-learn/stable/modules/generated/sklearn.preprocessing.StandardScaler.html (accessed Oct. 09, 2022).  
[10] A. Jung, Machine Learning: The Basics. Singapore, 2022.  
[11] ‘Multilayer Perceptron - an overview | ScienceDirect Topics’. [Online]. Available:  
https://www.sciencedirect.com/topics/computer-science/multilayer-perceptron (accessed Oct.
09, 2022).  
[12] ‘sklearn.linear_model.LinearRegression’, scikit-learn. [Online]. Available:
https://scikit-learn/stable/modules/generated/sklearn.linear_model.LinearRegression.html
(accessed Oct. 09, 2022).  
[13] ‘sklearn.neural_network.MLPRegressor’, scikit-learn. [Online]. Available:
https://scikit-learn/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
(accessed Oct. 09, 2022).  
[14] ‘sklearn.metrics.mean_absolute_error’, scikit-learn. [Online]. Available:
https://scikit-learn/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
(accessed Oct. 09, 2022).  
[15] ‘Train Test Validation Split: How To & Best Practices [2022]’. [Online]. Available:
https://www.v7labs.com/blog/train-validation-test-set (accessed Oct. 09, 2022).  
