#apidocs

import pandas as pd 
import numpy as np  
from flask import Flask, request, render_template
from flasgger import  Swagger
import pickle 

app = Flask(__name__)
Swagger(app)
pickle_in = open("model.pkl", "rb")
classifier = pickle.load(pickle_in)
pickle_in.close()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    """Loan approval
    ---
    parameters:
        - name: Married
          in: query
          type: number
          requried: true
        - name: Education
          in: query
          type: number
          requried: true
        - name: Credit_History
          in: query
          type: number
          requried: true
        - name: Income
          in: query
          type: number
          requried: true
    responses:
        200:
            description: The output values
    """
    f = [int(x) for x in request.args.values()]
    Married, Education, Credit_History, Income = f
    output = classifier.predict([[Married, Education, Credit_History, Income]])
    if output[0]==1:
        return 'Loan has been approved'
    return "Loan has not be approved"

@app.route("/predict_file",methods=["POST"])
def predict_file():
    """Loan Approval
    ---
    parameters:
        - name: file
          in: formData
          type: file
          requried: true
    responses:
        200:
            description: The output values
    """
    df = pd.read_csv(request.files.get("file"))
    output = classifier.predict(df)
    return str(list(output))


if __name__ == "__main__":
    app.run(port=8080, host="0.0.0.0")
