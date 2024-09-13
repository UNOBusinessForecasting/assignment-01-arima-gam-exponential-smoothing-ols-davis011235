import pandas as pd
from prophet import Prophet

data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/econ8310-assignment1/main/assignment_data_train.csv")

prophdat = data[['Timestamp', 'trips']]
prophdat.Timestamp = pd.to_datetime(prophdat.Timestamp)
prophdat = pd.DataFrame(prophdat.values, columns = ['ds','y'])

model = Prophet()
modelFit = model.fit(prophdat)
future = modelFit.make_future_dataframe(freq = 'H', periods=744)
forecast = modelFit.predict(future)

pred = list(modelFit.predict(future)['trend'])
 
print(pred)

