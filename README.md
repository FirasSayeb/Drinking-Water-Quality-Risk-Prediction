# Water Quality Risk Prediction

This project is a web application that predicts whether water is safe to drink based on several chemical and physical parameters. It uses a **TensorFlow neural network model** and a **Flask web interface** to allow users to input water quality data and receive a safety prediction with confidence probability.

---

## Project Structure


water_quality_prediction/
│
├── templates/
│ └── index.html # Frontend form for user input
│
├── water_potability.csv # Dataset containing water quality parameters and labels
│
├── main.py # Flask app with TensorFlow model
│
├── demo.mp4 # Demo video showing the app in action
│
└── README.md # This file


---

## Features

- Predict water safety (Safe / Not Safe) based on 5 key parameters:  
  - pH  
  - Turbidity  
  - Chloramines  
  - Conductivity  
  - Organic Carbon   
- Handles **missing inputs** by using column averages from the dataset  
- Professional, responsive web UI with **Bootstrap 5**  
- Model trained with **Dropout** and **class weighting** for improved accuracy  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/water_quality_prediction.git
cd water_quality_prediction

Create a virtual environment:

python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

Install dependencies:

pip install -r requirements.txt

Dependencies:

Flask

TensorFlow

Pandas

NumPy

scikit-learn

Bootstrap (via CDN in HTML)

Usage

Run the Flask app:

python main.py

Open your browser and go to:

http://127.0.0.1:5000/

Fill in the form with water parameters: 

pH

Turbidity

Chloramines

Conductivity

Organic Carbon

Click Predict to see whether the water is safe or not, along with the confidence probability.

Demo

Check out the demo of the web app in action:

demo.mp4 – Shows entering sample water parameters and receiving a prediction.

Dataset

water_potability.csv contains 9 water quality parameters for each sample and a target column Potability (0 = Not Safe, 1 = Safe).

Missing values are filled using column averages.

Data is scaled using StandardScaler before feeding into the TensorFlow model.

Model Details

Architecture:

Input: 9 features

Hidden Layer 1: 64 neurons, ReLU, Dropout 30%

Hidden Layer 2: 64 neurons, ReLU, Dropout 30%

Output: 1 neuron, Sigmoid (Safe/Not Safe)

Training:

Optimizer: Adam

Loss: Binary Crossentropy

Epochs: 100

Batch size: 32

Class weights to handle imbalance

Prediction: Outputs probability of being safe water.

Screenshots / UI

User inputs parameters in a clean Bootstrap card form.

Prediction appears below the form in a colored alert box.