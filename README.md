# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the data and use label encoder to change all the values to numeric.
2. Drop the unwanted values
3. Check for NULL values
4. Duplicate values.
5. Classify the training data and the test data.
6. Calculate the accuracy score, confusion matrix and classification report.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: JANANI S
RegisterNumber:  212223230086
*/
```
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])

data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
## Output:

![Screenshot 2024-10-13 182840](https://github.com/user-attachments/assets/a851c61a-53d0-401a-a1e4-c816830f1815)
![Screenshot 2024-10-13 182845](https://github.com/user-attachments/assets/3ed83a85-5d6b-4a17-a2b6-2a5431a029a3)
![Screenshot 2024-10-13 182850](https://github.com/user-attachments/assets/79d43654-0a83-4bd6-bf7e-7db48ffefb95)
![Screenshot 2024-10-13 182855](https://github.com/user-attachments/assets/4839946e-1456-423d-9a04-fbfda2a09687)
![Screenshot 2024-10-13 182859](https://github.com/user-attachments/assets/eea68273-af8d-4a05-847d-80fe324d10b4)
![Screenshot 2024-10-13 182904](https://github.com/user-attachments/assets/b2cb7669-7496-4914-bd5b-e17d73a16ab0)

![Screenshot 2024-10-13 182910](https://github.com/user-attachments/assets/42bae658-366b-49a4-93dd-aeb01cbce9fd)
![Screenshot 2024-10-13 182917](https://github.com/user-attachments/assets/78bd0484-829c-4e45-a25d-bdf75e3c732e)
![Screenshot 2024-10-13 182922](https://github.com/user-attachments/assets/2fe557ae-0109-40fb-ace5-6bfd9e0767c2)
![Screenshot 2024-10-13 182932](https://github.com/user-attachments/assets/f49dfc7f-a0ec-461c-927c-82f4f7b7d6ae)
![Screenshot 2024-10-13 182937](https://github.com/user-attachments/assets/a11c5ff6-4b84-4cc6-a64f-15cbf87c5f11)
![Screenshot 2024-10-13 182941](https://github.com/user-attachments/assets/e347de17-adaf-4e30-8ddc-388e27fc2e64)
![Screenshot 2024-10-13 182945](https://github.com/user-attachments/assets/915b09d0-67c9-4519-8d8d-a6199fd9acfd)
![Screenshot 2024-10-13 182952](https://github.com/user-attachments/assets/393298b6-5ca0-4ee3-9d66-50f554ccc43e)
![Screenshot 2024-10-13 182957](https://github.com/user-attachments/assets/44c38698-3c75-4b87-900d-8967d49e885a)
![Screenshot 2024-10-13 183007](https://github.com/user-attachments/assets/d22e1cea-fdcb-4267-bd66-1a989cb6c6cd)
![Screenshot 2024-10-13 183432](https://github.com/user-attachments/assets/02eff71a-ec75-49f6-a607-b2048d28bbcf)
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
