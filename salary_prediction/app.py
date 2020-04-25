import numpy as np
from flask import Flask, render_template, jsonify, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))  

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ['POST'])
def predict():
    X_test = np.array([int(x) for x in request.form.values()]).reshape(-1,1)
    y_pred = model.predict(X_test)
    
    output = np.round(y_pred[0],2)
    return render_template(f"index.html",
                             prediction_test = f"Salary according to your experience should be ${output}")
 
if __name__ == "__main__":
    app.run(debug=True)

