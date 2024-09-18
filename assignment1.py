from pygam import LinearGAM, s,f,l
import pandas as pd

data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/econ8310-assignment1/main/assignment_data_train.csv")
new_data = pd.read_csv("https://raw.githubusercontent.com/UNOBusinessForecasting/assignment-01-arima-gam-exponential-smoothing-ols-davis011235/refs/heads/main/assignment_data_test.csv")
new_data = new_data[['month', 'day', 'hour']]

data.head()
x= data[[ 'month', 'day','hour']]
y = data['trips']

model = LinearGAM(s(0) + s(1) + s(2))
modelFit = model.gridsearch(x.values,y)

pred = modelFit.predict(new_data)

print(pred[-20:])
