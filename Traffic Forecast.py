# Libraries
import pandas as pd 
import warnings
import numpy as np
# Packages
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data=pd.read_csv("D:/Final YearProject 2ND copy/18.Efficacy of Bluetooth-Based Data collection for Road Traffic analysis and visualization using Big data Analytics/Source code/Traffic Forecast.py")
print("<----Dataset----->")
print("******************")
print(data.head(20))
print()


warnings.filterwarnings("ignore")

print("<------Preprocessing------->")
print("****************************")
data1=(data.isnull().sum())
print(data1)
print()

print("<------Remove Unwanted Column------>")
data=data.drop(["DateTime"],axis=1)
print(data.head(20))
print()

# Splitting the data into features and target sets
print("<--------Data Splitting--------->")
print("*********************************")
    #Splitting the values into xand y
x=data.drop("Vehicles",axis=1)
y=data["Vehicles"]
print("Data Splitted into x and y")
print("X Label Dataset")
print(x.head(20))
print("Y Label Dataset")
print(y.head(20))   
print()

print("<--------Data Splitting--------->")
print("*******************************")

 # splitting the data into the training and test set.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)
print()

x1_train=np.expand_dims(x_train, axis=2)

y1_train=np.expand_dims(y_train,axis=1)

x1_test=np.expand_dims(x_test,axis=2)

print("Algorithm implemntation")
print("***********************")
print("LSTM")
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x1_train.shape[1], 1)))

model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mae', metrics=['accuracy'])

model.fit(x1_train, y1_train, epochs = 10, batch_size = 32)

predictions = model.predict(x1_test)
print()

print("Classification Report")
print("Results of sklearn.metrics:")
print("***************************")
mae = metrics.mean_absolute_error(y_test,predictions)
mse = metrics.mean_squared_error(y_test,predictions)
rmse = np.sqrt(mse) #mse**(0.5)  
r2 = metrics.r2_score(y_test,predictions)

print("MAE:",mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-Squared:", r2)
print()

print("Linear Regression")
regressor = LinearRegression()  
regressor.fit(x_train, y_train) #training the algorithm

#To retrieve the intercept:
print(regressor.intercept_)

#For retrieving the slope:
print(regressor.coef_)

y_pred = regressor.predict(x_test)

print("Classification Report")
print("Results of sklearn.metrics:")
print("***************************")
mae1 = metrics.mean_absolute_error(y_test,y_pred)
mse1 = metrics.mean_squared_error(y_test,y_pred)
rmse1 = np.sqrt(mse1) #mse**(0.5)  
r2 = metrics.r2_score(y_test,y_pred)
print()

print("MAE:",mae1)
print("MSE:", mse1)
print("RMSE:", rmse1)
print("R-Squared:", r2)

print("<------Prediction Status-------->")
print("*********************************")

data12=np.array([1,20151102221]).reshape(1,-1)

predictions=regressor.predict(data12)
abs(predictions)
if predictions>=20:
    print("Traffic")
else:
    print("Normal")
print()
        
print("<-------Comparision Graph------->")
print("********************************")
    #Comparision Graph b/w two algo
vals=[mse,mse1]
inds=range(len(vals))
labels=["LSTM ","Linear Regression" ]
fig,ax = plt.subplots()
rects = ax.bar(inds, vals)
ax.set_xticks([ind for ind in inds])
ax.set_xticklabels(labels)
plt.title('Comparison graph For LSTM & Regression')
plt.show()

import pickle
filename = 'traffic.pkl'
pickle.dump(regressor, open(filename, 'wb'))



























