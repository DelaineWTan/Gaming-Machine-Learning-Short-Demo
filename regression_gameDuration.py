import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
dataframe = pd.read_csv("dataset/games.csv")

# Drop rows with missing values in the target variable
dataframe.dropna(subset=['gameDuration'], inplace=True)

# Manually specify the columns you deem insignificant and keywords to exclude
columns_to_exclude = ["gameId", "creationTime", "seasonId", "winner"]
keywords_to_exclude = ["sum", "ban", "champ"]

# Drop the columns from the dataset
X = dataframe.drop(columns=["gameDuration"] + columns_to_exclude +
                   [col for col in dataframe.columns if any(keyword in col for keyword in keywords_to_exclude)])

y = dataframe["gameDuration"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

import statsmodels.api as sm

# Add a constant term to the features
X_train_const = sm.add_constant(X_train)

# Fit the OLS model (Ordinary Least Squares)
ols_model = sm.OLS(y_train, X_train_const).fit()

# Print a summary of the model
print(ols_model.summary())

# Evaluate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
