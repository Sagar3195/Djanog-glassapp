##Identify type of glasses based on the chemical composition using classification algorithm
## Importing required libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

##Loading dataset
data = pd.read_csv("glass.csv")

print(data.head())
print(data.shape)
print(data.info())
print('-----------------')
print(data.describe())

##Let's check missing values in dataset
print(data.isnull().sum())

#### We can see that there is no missing values in dataset

##We have 6 unique type of glasses
print(data['Type'].unique())

##Splitting dataset into independent variable and dependent variable
X = data.iloc[:,:-1]
y = data.iloc[:, -1]

#print(X.head())
#print(y.head())

##Feature selection
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)

print(model.feature_importances_)

#Let's visualize the feature importance
features = pd.Series(model.feature_importances_, index = X.columns)
features.nlargest(9).plot(kind = 'barh')
plt.title("Important features")
#plt.show()

##From above the plot we can see that all features are important for the model.

## Now splitting dataset into training data and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 42)
print(x_train.shape, x_test.shape)
print("==========")
print(y_train.shape, y_test.shape)

## Features scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

##Now we use random forest algorthim for classification
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=300, criterion = 'entropy', random_state= 42)

#Now train the model using fit method
classifier.fit(x_train, y_train)

#Now predict the model
prediction = classifier.predict(x_test)

#print(prediction)

##Let's check the performance of the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, prediction)
print("Accuracy of the model: ", accuracy)

print(confusion_matrix(y_test, prediction))

print(classification_report(y_test, prediction))

#import joblib
#joblib.dump(classifier, "glass_model.pkl")
