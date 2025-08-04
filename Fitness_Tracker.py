import pandas as pd
# creating dictionary for input of sample dataset
sample_data = {
    'Age' : [25, 45, 30, 50, 35, 23],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male'],
    'Weight_in_kg' : [68, 70, 55, 85, 60, 75],
    'Height_in_cm' : [175, 160, 165, 170, 158, 180],
    'Sleeping_hours_avg' : [7, 5, 8, 4, 6, 7],
    'Balanced_Diet' :  ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes'],
    'Exercise_type' : ['Daily', 'None', 'Weekly', 'None', 'Daily', 'Weekly'],
    'Weekly_exercise_Hours': [7, 0, 3, 0, 6, 4],
    'Is_Fit' : [1, 0, 1, 0, 1, 1] 
   }
datas = pd.DataFrame(sample_data) #dataframe
# print("Initial data: \n ",datas) #prints the table of data created, print(datas.to_string(index=False)) for full data print
datas['Gender'] = datas['Gender'] .map({'Male' : 0, 'Female' : 1})
datas['Balanced_Diet'] = datas['Balanced_Diet'] .map({'No' : 0, 'Yes' : 1})
datas['Exercise_type'] = datas['Exercise_type'] .map({'None' : 0, 'Daily' : 1, 'Weekly': 2})
datas['BMI'] = datas['Weight_in_kg'] / ((datas['Height_in_cm']/100) **2)
print("Initial data: \n ",datas)

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
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy: ", accuracy_score(y_test, y_pred)) #checking performance
print("Classification report: \n", classification_report(y_test, y_pred))

#getting input
print("Enter the following details:\n")
age = int(input("Age: "))
gender = 1 if input("Gender (Male/Female): ").lower() == "male" else 1
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
