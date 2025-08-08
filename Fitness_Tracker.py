import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

datas = pd.read_csv("fitness_data_synthetic_200.csv") #reading the csv file
#Data inspection:
print(datas.head())
print(datas.info())
print(datas.isnull().sum())
print(datas.describe())

#data mapping manually
datas['Gender'] = datas['Gender'].map({'Male':0, 'Female':1})
datas['Balanced_Diet'] = datas['Balanced_Diet'].map({'No':0, 'Yes':1})
datas['Exercise_type'] = datas['Exercise_type'].map({'None':0, 'Weekly':1, 'Daily':2})
print(datas.head()) #checks if the mapping is done right
datas['BMI'] = datas['Weight_in_kg'] / ((datas['Height_in_cm'] /100) **2)
print(datas.isnull().sum()) # for checking for missing values


#features and label
x = datas[['Age', 'Gender', 'Weight_in_kg','Height_in_cm', 'Sleeping_hours_avg', 'Balanced_Diet', 'Exercise_type', 'Weekly_exercise_Hours', 'BMI']]
y = datas['Is_Fit']

#split data to train the model
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state= 42)

# using random classifier to train the data
from sklearn.ensemble import RandomForestClassifier
clfr = RandomForestClassifier(n_estimators= 100, random_state= 42)
clfr.fit(x_train, y_train)

#testing the model based on accuracy, precision, recall, F1( balance btw p,a)
y_pred = clfr.predict(x_test) #predicting
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy: ", accuracy_score(y_test, y_pred)) #checking performance
print("Classification report: \n", classification_report(y_test, y_pred))

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("This is confusion matrix:", cm)

#Heatmap for cm
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Fit', 'Fit'],
            yticklabels=['Not Fit', 'Fit'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

#getting input
print("Enter the following details:\n")
print("\nEnter the following details:")
while True:
    try:
        age = int(input("Age: "))
        break
    except ValueError:
        print("Please enter a valid number for age.")
gender = 0 if input("Gender (Male/Female): ").lower() == "male" else 1
weight = float(input("Weight (kg): "))
height = float(input("Height (cm): "))
sleep = float(input("Sleep (hours): "))
diet = 1 if input("Balanced Diet? (Yes/No): ").lower() == "yes" else 0
exercise_type_input = input("Exercise frequency (None/Weekly/Daily): ").strip().lower()
exercise_type_map = {'none': 0, 'weekly': 1, 'daily': 2}
exercise_type = exercise_type_map.get(exercise_type_input, 0)
if exercise_type in [1, 2]:
    weekly_exercise = float(input("How many hours do you exercise per week? "))
else:
    weekly_exercise = 0.0
bmi = weight / ((height / 100) ** 2)

#converting input to same format as that of the already created dataset
user = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Weight_in_kg': [weight],
    'Height_in_cm': [height],
    'Sleeping_hours_avg': [sleep],
    'Balanced_Diet': [diet],
    'Exercise_type': [exercise_type],
    'Weekly_exercise_Hours': [weekly_exercise],
    'BMI': [bmi]
})

#executing it
result = clfr.predict(user)
print("You are FIT. Continue your healthy lifestyle." if result[0] == 1 else "You are NOT FIT. Kindly change your lifestyle to be more healthier.")
