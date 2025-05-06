from flask import Flask, request, render_template
import pandas as pd
import pickle
import sqlite3  # Built-in sqlite3 module

app = Flask(__name__)

# Load the model
with open('insurance_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize the database
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY, age REAL, sex TEXT, bmi REAL, children INTEGER, smoker TEXT, region TEXT, prediction REAL)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Capture form input
        int_features = [x for x in request.form.values()]
        print("Received input features:", int_features)
        
        # Create DataFrame
        final_features = pd.DataFrame([int_features], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
        print("Initial DataFrame:\n", final_features)

        # Ensure correct data types
        final_features['age'] = final_features['age'].astype(float)
        final_features['bmi'] = final_features['bmi'].astype(float)
        final_features['children'] = final_features['children'].astype(float)

        # Process the categorical features the same way they were during training
        categorical_features = ['sex', 'smoker', 'region']
        numerical_features = ['age', 'bmi', 'children']
        
        sample_data = final_features[numerical_features + categorical_features]

        # Align columns to match the training data
        X_train_columns = list(model.named_steps['preprocessor'].transformers_[0][2]) + list(model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))
        
        # Transform sample data
        sample_data_transformed = model.named_steps['preprocessor'].transform(sample_data)
        sample_data_transformed = pd.DataFrame(sample_data_transformed, columns=X_train_columns)
        print("Transformed DataFrame:\n", sample_data_transformed)

        # Predict
        prediction = model.named_steps['regressor'].predict(sample_data_transformed)[0]
        print("Prediction:", prediction)

        # Save the result to the database
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute("INSERT INTO predictions (age, sex, bmi, children, smoker, region, prediction) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (final_features['age'][0], final_features['sex'][0], final_features['bmi'][0], final_features['children'][0], final_features['smoker'][0], final_features['region'][0], prediction))
        conn.commit()
        conn.close()

        return render_template('index.html', prediction_text=f'Predicted Insurance Charge: {prediction:.2f}', 
                               age=request.form['age'], 
                               sex=request.form['sex'], 
                               bmi=request.form['bmi'], 
                               children=request.form['children'], 
                               smoker=request.form['smoker'], 
                               region=request.form['region'])
    except Exception as e:
        print("Exception occurred:", e)
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
