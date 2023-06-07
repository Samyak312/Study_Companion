import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("studentdata.csv")  # Replace "your_dataset.csv" with the actual filename

# Split the dataset into features (X) and target variable (y)
X = data[['Time Spent (minutes)', 'Quiz Score', 'Practice Problems Solved']]
y = data['Topic']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate mean values as references
mean_time_spent = X_train['Time Spent (minutes)'].mean()
mean_quiz_score = X_train['Quiz Score'].mean()
mean_practice_problems = X_train['Practice Problems Solved'].mean()

# Replace missing values with the mean values
X_train['Time Spent (minutes)'].fillna(mean_time_spent, inplace=True)
X_train['Quiz Score'].fillna(mean_quiz_score, inplace=True)
X_train['Practice Problems Solved'].fillna(mean_practice_problems, inplace=True)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Predict the topics for the test set
X_test['Time Spent (minutes)'].fillna(mean_time_spent, inplace=True)
X_test['Quiz Score'].fillna(mean_quiz_score, inplace=True)
X_test['Practice Problems Solved'].fillna(mean_practice_problems, inplace=True)
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Get the feature importances
feature_importances = clf.feature_importances_
print("Feature Importances:", feature_importances)

# Determine the weak and strong topics for each student in the test set
student_topics = data.loc[X_test.index]['Topic']
for i, student_topic in enumerate(student_topics):
    if y_pred[i] == student_topic:
        print(f"Student {i+1}: Strong in {student_topic}")
    else:
        print(f"Student {i+1}: Weak in {student_topic}")