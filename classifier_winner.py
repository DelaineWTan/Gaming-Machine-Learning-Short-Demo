import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from replace_champion_and_summoner_names import format_dataframe

# Load the dataset
dataframe = pd.read_csv("dataset/games.csv")

# Set the maximum number of columns to display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)

# Skipped preprocessing as it was unnecessary  (handle missing values, encode categorical
# variables, etc.)
# For example:
# data.dropna(inplace=True)
# data = pd.get_dummies(data)

# Manually specify the columns you deem insignificant
columns_to_exclude = ["gameId", "creationTime", "gameDuration", "seasonId"]

# Drop the columns from the dataset
X = dataframe.drop(columns=["winner"] + columns_to_exclude)  # Drop the target variable as well
y = dataframe["winner"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000, solver="liblinear")
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Show coefficients and their corresponding feature names
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_[0]})
# print(coefficients)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
total_samples = len(y_test)
correct_predictions = sum(y_test == y_pred)
incorrect_predictions = total_samples - correct_predictions
print("#################### MODEL ACCURACY ####################")
print(f"Out of {total_samples} test samples:")
print(f"- Correct predictions: {correct_predictions}")
print(f"- Incorrect predictions: {incorrect_predictions}")
print(f"Accuracy: {accuracy:.2%}")

print("\n#################### PREDICTIONS ####################")
# Select 10 random rows from the original dataframe
new_data = dataframe.sample(n=10, axis=0, random_state=42)
print("10 Random Rows from Original Dataframe")
print(format_dataframe(new_data))
# Manually specify the columns you deem insignificant
columns_to_exclude = ["gameId", "creationTime", "gameDuration", "seasonId"]
# Drop the insignificant columns from the new dataset
X_new = new_data.drop(columns=["winner"] + columns_to_exclude)
# Make predictions on the new data using the trained model
y_pred_new = model.predict(X_new)

# Define a dictionary to map predicted labels to team names
team_mapping = {1: "Team 1", 2: "Team 2"}

# Map the predicted labels to team names
y_pred_new_readable = [team_mapping[label] for label in y_pred_new]

# Print the human-readable predictions
print("\nPredicted Winners:")
for i, prediction in enumerate(y_pred_new_readable):
    print(f"Match {i + 1}: {prediction}")

