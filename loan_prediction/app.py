import pandas as pd 
import numpy as np 
from flask import Flask, request, render_template

app = Flask(__name__)
import pickle 
pickle_in = open("model.pkl","rb")
classifier = pickle.load(pickle_in)
pickle_in.close()   

@app.route("/")
def welcome():
    return render_template("home.html")

@app.route("/predict")
def prediction():
  
    f = np.array([int(x) for x in request.args.get("features")]).reshape(1,4)
    output = classifier.predict(f) 
    if output[0]==1:
        return "Your Loan has be approved"
    return "Your Loan has not be approved"

@app.route("/predict_file", methods=["POST"])
def predict_file():
    df = pd.read_csv(request.files.get("file"))
    output = classifier.predict(df)
    return str(list(output))



if __name__ == "__main__":
    app.run(debug=True)  