# Creating the flask app

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the models
models = {
    'Linear Regression': joblib.load('models/linear.pkl'),
    'kNN': joblib.load('models/knn.pkl'),
    'SVM': joblib.load('models/svm.pkl'),
    'MLP': joblib.load('models/mlp.pkl')
}

# Get feature names from X columns
features = joblib.load('data/features.pkl')

# Load the dataframes
comparison_df = pd.read_csv('data/comparison_df.csv')
feature_importance_df = pd.read_csv('data/feature_importance_df.csv')
evaluation_df = pd.read_csv('data/evaluation_df.csv')

@app.route('/')
def home():
    comparison_table = comparison_df.to_html(classes='dataframe', header=True, index=False)
    feature_importance_table = feature_importance_df.to_html(classes='dataframe', header=True, index=False)
    evaluation_table = evaluation_df.to_html(classes='dataframe', header=True, index=False)
    return render_template('index.html', evaluation_table=evaluation_table, comparison_table=comparison_table, feature_importance_table=feature_importance_table, features=features, models=models.keys())  

@app.route('/predict', methods=['POST'])
def predict():
    selected_model = request.form['model']
    
    # Retrieve feature values from form data
    feature_values = []
    for feature in features:
        value = float(request.form[feature])
        feature_values.append(value)
    
    # Convert feature_values to numpy array
    features_array = np.array(feature_values).reshape(1, -1)

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_array)
    
    # Get the selected model
    model = models[selected_model]
    
    # Make prediction
    prediction = model.predict(scaled_features)
    
    comparison_table = comparison_df.to_html(classes='dataframe', header=True, index=False)
    feature_importance_table = feature_importance_df.to_html(classes='dataframe', header=True, index=False)
    evaluation_table = evaluation_df.to_html(classes='dataframe', header=True, index=False)
    return render_template(
        'index.html',
        prediction_text='Predicted Points: {:.2f}'.format(prediction[0]),
        features=features,
        models=models.keys(),
        evaluation_table=evaluation_table,
        comparison_table=comparison_table,
        feature_importance_table=feature_importance_table
    )
    #return render_template('index.html', prediction_text='Predicted Points: {:.2f}'.format(prediction[0]), features=features, models=models.keys(), comparison_table=comparison_table, feature_importance_table=feature_importance_table)

if __name__ == "__main__":
    app.run(debug=True)
