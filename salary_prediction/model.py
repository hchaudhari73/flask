import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("Salary_Data.csv")

X = np.array(df[df.columns[0]]).reshape(-1,1)
y = df[df.columns[1]]

model = LinearRegression()
model.fit(X,y)

pickle.dump(model, open("model.pkl","wb"))