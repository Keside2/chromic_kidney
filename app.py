import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template, redirect, url_for
from sklearn.metrics import recall_score, precision_score, f1_score

app = Flask(__name__)

# Loading and preprocessing the dataset
df = pd.read_csv('./data/kidney_disease.csv')
columns_to_retain = ['pcc', 'pc', 'rbc' 'age', 'bp','sg', 'al', 'sc', 'hemo', 'pcv', 'htn', 'classification']
df = df.drop([col for col in df.columns if col not in columns_to_retain], axis=1)
df = df.dropna(axis=0)

# Encode categorical columns
label_encoders = {}
for column in df.columns:
    if not pd.api.types.is_numeric_dtype(df[column]):
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

X = df.drop(['classification'], axis=1)
y = df['classification']
x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Training the RandomForest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate recall, precision, and F1-score
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(rf_model.predict(X_test))
print("The accuracy of this model is", rf_model.score(X_test, y_test) * 100)
print("Recall: {:.2f}".format(recall))
print("Precision: {:.2f}".format(precision))
print("F1-score: {:.2f}".format(f1))

# Define a mapping for categorical values
category_mapping = {
    'htn': {'yes': 1, 'no': 0},
    # Add other mappings if there are more categorical features
}

# Function to predict the class and probability of CKD
def predict_ckd(features):
    x = np.array(features).reshape(1, -1)
    x = x_scaler.transform(x)
    y_pred = rf_model.predict(x)
    proba = rf_model.predict_proba(x)
    return y_pred[0], proba.max()

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_redirect():
    return redirect(url_for('home_page'))

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/registerme', methods=['POST'])
def register_redirect():
    return redirect(url_for('home_page'))

@app.route('/home')
def home_page():
    return render_template('account.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form.to_dict()
    for feature in features:
        if feature in category_mapping:
            features[feature] = category_mapping[feature].get(features[feature].lower(), 0)
        else:
            features[feature] = float(features[feature])
    
    # Ensure the features are in the correct order
    feature_values = [features[column] for column in columns_to_retain if column != 'classification']
    
    pred, proba = predict_ckd(feature_values)
    if pred == 0:
        return render_template('account.html', prediction_text='The patient has CKD with probability {:.2f}%'.format(proba * 100))
    else:
        return render_template('account.html', prediction_text='The patient does not have CKD with probability {:.2f}%'.format(proba * 100))

if __name__ == '__main__':
    app.run(debug=True)
