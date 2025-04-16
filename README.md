# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the libraries and read the data frame using pandas

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Thirunavukkarasu meenakshisundaram
RegisterNumber: 212224220117  
*/
```

```
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```
## Output:

![Screenshot 2025-04-16 134241](https://github.com/user-attachments/assets/d2ce1bf0-2d46-43eb-bd95-6ab30b6f9319)

![Screenshot 2025-04-16 134224](https://github.com/user-attachments/assets/38fb6bb9-14c8-461e-9932-c9010be80c09)

![Screenshot 2025-04-16 134241](https://github.com/user-attachments/assets/dbf7a147-1ae3-4fc7-8e16-c337f07bc95e)

data.head() for salary

![Screenshot 2025-04-16 134241](https://github.com/user-attachments/assets/116587d2-eba0-4862-8fcb-a174ac862e2c)

MSE value

![Screenshot 2025-04-16 134611](https://github.com/user-attachments/assets/4d298922-a2df-474f-b7c1-0105eff64fc6)

r2 value

![Screenshot 2025-04-16 134643](https://github.com/user-attachments/assets/1671643d-6191-4ad5-91ee-dfd61c243651)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
