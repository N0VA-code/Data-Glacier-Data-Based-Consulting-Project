from flask import Flask, request, render_template, redirect, url_for
import joblib
import pandas as pd
import json

app = Flask(__name__)

# Load the model
model = joblib.load('model/gradient_boosting_model.joblib')

# Load the encoder
encoder = joblib.load('encoder.joblib')

# Load the column names for one-hot encoded features
with open('columns.json', 'r') as fh:
    model_columns = json.load(fh)

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from the form
    form_data = request.form.to_dict()
    numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    numerical_data = {key: float(form_data[key]) for key in numerical_columns}
    
    # One-hot encode the categorical data and convert to a DataFrame
    categorical_data = {key: [form_data[key]] for key in form_data if key not in numerical_columns}
    categorical_df = pd.DataFrame(categorical_data)
    categorical_encoded = encoder.transform(categorical_df)
    
    # Create a DataFrame for the encoded categorical features with the right column names
    categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_df.columns))
    
    # Create a DataFrame for the numerical features
    numerical_df = pd.DataFrame([numerical_data])
    
    # Combine the numerical and categorical dataframes
    features_df = pd.concat([numerical_df, categorical_encoded_df], axis=1)
    
    # Ensure the DataFrame has the same columns as the model expects
    features_df = features_df.reindex(columns=model_columns, fill_value=0)
    
    # Predict the probability
    probability = model.predict_proba(features_df)[0][1]
    
    # Redirect to the results page with the prediction text
    return redirect(url_for('results', prediction_text=f'Subscription Probability: {probability:.2%}'))

@app.route('/results')
def results():
    # Retrieve the prediction text from the query string
    prediction_text = request.args.get('prediction_text')
    return render_template('results.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
