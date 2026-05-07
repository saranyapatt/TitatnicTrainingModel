import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from dotenv import load_dotenv

load_dotenv()
new_path = os.getenv('PATH2TRAIN')
test_path = os.getenv('PATH2TEST')

data = pd.read_csv(new_path)
data_test_ori = pd.read_csv(test_path)
data_test = pd.read_csv(test_path)

data['title'] = data['Name'].str.extract(r',\s*([A-Za-z]+)\.', expand=False)
data_test['title'] = data_test['Name'].str.extract(r',\s*([A-Za-z]+)\.', expand=False)

title_mapping = {
    "Mr": "Mr",
    "Miss": "Miss_Mrs", "Mrs": "Miss_Mrs", "Mlle": "Miss_Mrs", "Ms": "Miss_Mrs", "Mme": "Miss_Mrs",
    "Master": "Master",
    "Lady": "Royalty", "Sir": "Royalty", "Jonkheer": "Royalty", "Don": "Royalty", "Countess": "Royalty",
    "Dr": "Officer", "Rev": "Officer", "Col": "Officer", "Major": "Officer", "Capt": "Officer"
}

data['title_group'] = data['title'].map(title_mapping)
data_test['title_group'] = data_test['title'].map(title_mapping)

title_dummies = pd.get_dummies(data['title_group'], prefix='Title', dtype=int)
title_dummies_test = pd.get_dummies(data_test['title_group'], prefix='Title', dtype=int)
data = pd.concat([data, title_dummies], axis=1)
data_test = pd.concat([data_test, title_dummies_test], axis=1)
data_test = data_test.reindex(columns=data.columns, fill_value=0)


data = data.drop(['Name', 'title', 'title_group'], axis=1)
data_test = data_test.drop(['Name', 'title', 'title_group'], axis=1)

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data_test['Sex'] = data_test['Sex'].map({'male': 0, 'female': 1})

data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data_test['Embarked'] = data_test['Embarked'].fillna(data_test['Embarked'].mode()[0])

embarked_dummies = pd.get_dummies(data['Embarked'], prefix='Embarked', dtype=int)
embarked_dummies_test = pd.get_dummies(data_test['Embarked'], prefix='Embarked', dtype=int)

data = pd.concat([data, embarked_dummies], axis=1)
data_test = pd.concat([data_test, embarked_dummies_test], axis=1)
data.drop('Embarked', axis=1, inplace=True)
data_test.drop('Embarked', axis=1, inplace=True)

titles = ['Title_Mr', 'Title_Miss_Mrs', 'Title_Master', 'Title_Royalty', 'Title_Officer']
for title in titles:
    median_val = data[data[title] == 1]['Age'].median()
    data.loc[(data['Age'].isnull()) & (data[title] == 1), 'Age'] = median_val

data['Age'] = data['Age'].fillna(data['Age'].median())

for title in titles:
    median_val = data_test[data_test[title] == 1]['Age'].median()
    data_test.loc[(data_test['Age'].isnull()) & (data_test[title] == 1), 'Age'] = median_val

    median_val_fare = data_test[data_test[title] == 1]['Fare'].median()
    data_test.loc[(data_test['Fare'].isnull()) & (data_test[title] == 1), 'Fare'] = median_val_fare
data_test['Age'] = data_test['Age'].fillna(data_test['Age'].median())
data_test['Fare'] = data_test['Fare'].fillna(data_test['Fare'].median())

data.drop('Cabin', axis=1, inplace=True)
data_test.drop('Cabin', axis=1, inplace=True)

data.drop('Ticket', axis=1, inplace=True)
data_test.drop('Ticket', axis=1, inplace=True)

data.drop('PassengerId', axis=1, inplace=True)
data_test.drop('PassengerId', axis=1, inplace=True)

data_y = data['Survived']
data.drop('Survived', axis=1, inplace=True)
data_test.drop('Survived', axis=1, inplace=True)
scaler = StandardScaler()

scaler.fit(data)
X_train = scaler.transform(data)
X_test = scaler.transform(data_test)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, data_y)
y_pred = classifier.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': data_test_ori['PassengerId'],
    'Survived': y_pred
})

submission.to_csv('submission.csv', index=False)