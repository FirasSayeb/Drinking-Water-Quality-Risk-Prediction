import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

my_csv = Path("water_potability.csv")
data = pd.read_csv(my_csv.resolve())

data = data.apply(pd.to_numeric, errors="coerce")
data.fillna(data.mean(), inplace=True)

feature_columns = data.drop("Potability", axis=1).columns

X = data.drop("Potability", axis=1).values
y = data["Potability"].values

column_means = data.drop("Potability", axis=1).mean().values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(9,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    ph = float(request.form['ph'])
    turbidity = float(request.form['turbidity'])
    chloramines = float(request.form['chloramines'])
    conductivity = float(request.form['conductivity'])
    organic_carbon = float(request.form['organic_carbon'])

    input_data = column_means.copy()

    input_data[0] = ph
    input_data[3] = chloramines
    input_data[5] = conductivity
    input_data[6] = organic_carbon
    input_data[8] = turbidity

    input_data = input_data.reshape(1, -1)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    result = "Safe Water ✅" if prediction[0][0] > 0.5 else "Not Safe ❌"

    return render_template('index.html', prediction_text=result)


if __name__ == '__main__':
    app.run(debug=True)