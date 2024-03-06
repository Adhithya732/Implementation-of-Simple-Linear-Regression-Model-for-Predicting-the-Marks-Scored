# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Predict the regression for marks by using the representation of the graph
4. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Adhithya A
RegisterNumber:  212222220004
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

### Dataset:
![278995575-c7816d33-6dab-45e2-8d19-9a11e9583cb5](https://github.com/Adhithya732/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/162408377/476f7180-40d3-4c65-8ca9-87a83688b174)taset:

### Head values:
![278996479-7f3d7783-4601-4e70-989f-2ccbf87d0765](https://github.com/Adhithya732/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/162408377/fd4cce78-fb50-48a4-b16f-ce5775a854af)

### Tail values:
![278996533-5343e114-fe3a-4ad7-8058-6b81db462fdc](https://github.com/Adhithya732/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/162408377/fb9f0dd8-0cff-4ccf-bcd1-1ed21b0cfdf0)

### X and Y values:
![278996577-f84947e0-99a3-444c-8286-c59cc0660a4e](https://github.com/Adhithya732/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/162408377/e924a9a9-7a74-4b41-b99f-83e86cf6fc75)

### Predication values of X and Y:
![278996605-6ea46100-8530-4491-821e-079308a1eef5](https://github.com/Adhithya732/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/162408377/0aabf1f6-5ed5-4c3b-b616-21430fa62011)
### MSE,MAE and RMSE:
![image](https://github.com/Adhithya732/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/162408377/9a0ca3d1-2230-4335-9fd3-aaefe57fac23)

### Training Set:
![278996909-088c3714-a70d-4ef0-b952-1d26c48e1fa8](https://github.com/Adhithya732/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/162408377/7aa245eb-9f79-4930-a72c-889855b9d3c3)
### Testing Set:
![278996855-aa18e6a5-11f7-410e-bbd6-89c052ff52a6](https://github.com/Adhithya732/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/162408377/77cc7dc7-dad1-4755-8567-9541c70722b8)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
