A web app which allows the user to predict the points scoring output of a player given other statistics of that player (e.g. Rebounds, Assists, FG%...). 
The user can select the machine learning model he wants to use for the prediction. There are 4 model classes: linear regression, knn, svm, mlp.
The dataset used to train, tune the hyperparameter and test the models is: https://www.kaggle.com/datasets/amirhosseinmirzaie/nba-players-stats2023-season.
For hyperparameter tuning, I used a grid search method. For each model the parameters optimized are different, due to the different nature of the models themselves. You can find more information on the optimized parameters in the power point presentation.
The models are evaluated with the following metrics: MAE, MSE, R-squared. MLP is the strongest model by any metric considered. On the other hand, MLP is also the slowest model at training time.
