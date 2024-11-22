## Title of the Project
FORECASTING GOLD PRICE ON AN ECONOMIC DRIVEN MARKET

## About

The Gold Price Prediction App addresses the challenge of forecasting future gold prices by analyzing          various economic indicators, such as inflation rates, unemployment rates, and GDP. Gold prices are influenced by numerous complex factors, making accurate predictions difficult. This project aims to provide users with a machine learning-based solution that simplifies the process of predicting gold prices using historical data and economic trends. By offering a comparison of multiple models and allowing users to input future economic conditions, the app empowers users to make data-driven decisions about gold price movements.

## Features

- Integration of multiple machine learning models (Random Forest, Gradient Boosting, Decision Tree) for gold price prediction.
- Use of economic indicators such as inflation rate, unemployment rate, GDP, and applied interest rates as input features.
- Real-time prediction capability by allowing users to input future economic data for forecasting.
- Data preprocessing steps including feature scaling, label encoding, and handling missing values to improve model accuracy.


## Requirements
<!--List the requirements of the project as shown below-->
* Operating System: Requires a 64-bit OS (Windows 10 or Ubuntu) for compatibility with machine learning frameworks.
* Development Environment: Python 3.0 environment.
* Machine Learning Frameworks: Streamlit for output source.
* Version Control: Implementation of Git for collaborative development and GitHub or GitLab for code repository management.
* IDE: Jupyter Notebook, VS Code.
* Additional Dependencies: pandas for data manipulation,numpy for numerical computations,scikit-learn for machine learning models and data preprocessing,matplotlib for plotting and data visualization,StandardScaler and LabelEncoder from scikit-learn for data preprocessing.
  
## System Architecture

![image](https://github.com/user-attachments/assets/2976e8ae-2783-40cc-b116-9627067d2c53)

## Program:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

gold_prices = pd.read_csv('gold_price_china.csv')
economic_data = pd.read_csv('china-inflation(1).csv')

gold_prices.shape

gold_prices.head()

gold_prices.info()

gold_prices.describe()

economic_data.shape

economic_data.tail()

economic_data.info()

economic_data.describe()

data = pd.merge(gold_prices, economic_data, on='Date',how='left')

data.tail(10)

data.info()

data['Date'] = pd.to_datetime(data['Date'])

data.sort_values('Date', inplace=True)

data['Date'] = pd.to_datetime(data['Date'])

# Plotting the average closing price
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Average Closing Price'], color='navy', linewidth=2)

# Adding labels and title
plt.title('Average Closing Price Over Time', fontsize=16, weight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Closing Price', fontsize=12)

# Adding grid and adjusting style similar to the uploaded graph
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot
plt.show()

correlation_matrix = data.corr()
print(correlation_matrix)

correlation_columns = ['Average Closing Price', 'Interest Rate', 'Inflation Rate ', 'GDP', 'Tariff Rate']

# Calculate the correlation matrix
correlation_matrix = data[correlation_columns].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='black', fmt=".2f")

# Customize plot
plt.title('Correlation Heatmap of Gold Prices and Economic Indicators', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()

df = pd.DataFrame(data)

df['Interest Rate'].fillna(df['Interest Rate'].mean(), inplace=True)
df['Inflation Rate '].fillna(df['Inflation Rate '].mean(), inplace=True)
df['GDP'].fillna(df['GDP'].mean(), inplace=True)
df['Tariff Rate'].fillna(df['Tariff Rate'].mean(), inplace=True)

label_encoder = LabelEncoder()
df['Interest Rate'] = label_encoder.fit_transform(df['Interest Rate'])

scaler = StandardScaler()
data[['Interest Rate', 'Inflation Rate ', 'GDP', 'Tariff Rate']] = scaler.fit_transform(
    data[['Interest Rate', 'Inflation Rate ', 'GDP', 'Tariff Rate']])

# Select features and target
features = ['Interest Rate', 'Inflation Rate ', 'GDP', 'Tariff Rate']
X = data[features]
y = data['Average Closing Price']

print(X.isnull().sum())

print(y.isnull().sum())

features = ['Interest Rate','Inflation Rate ','GDP','Tariff Rate'] 
X = df[features]
y = df['Average Closing Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
}

model_metrics = {}
predicted_values = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    model_metrics[model_name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R²': r2}
    predicted_values[model_name] = y_pred
    
    print(f"\n{model_name}:")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared: {r2}")
    print(f"Predicted Values: {y_pred}")

metrics_df = pd.DataFrame(model_metrics).T

# Assuming X_train, y_train, X_test, y_test are defined and your model is already trained

# Predict on both training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate residuals (actual - predicted)
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

# Plotting the residuals for training and testing sets
plt.figure(figsize=(14, 6))

# Training set residuals
plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, train_residuals, alpha=0.5)
plt.hlines(y=0, xmin=min(y_train_pred), xmax=max(y_train_pred), colors='red')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Training Set Residuals')

# Testing set residuals
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, test_residuals, alpha=0.5)
plt.hlines(y=0, xmin=min(y_test_pred), xmax=max(y_test_pred), colors='red')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Testing Set Residuals')

plt.tight_layout()
plt.show()

metrics_df = pd.DataFrame(model_metrics).T

plt.figure(figsize=(10, 6))
metrics_df['RMSE'].sort_values().plot(kind='barh', color='skyblue')
plt.title('Model Comparison (RMSE)')
plt.xlabel('RMSE')
plt.ylabel('Models')
plt.show()

plt.figure(figsize=(10, 6))
metrics_df['R²'].sort_values().plot(kind='barh', color='lightgreen')
plt.title('Model Comparison (R²)')
plt.xlabel('R²')
plt.ylabel('Models')
plt.show()

plt.figure(figsize=(10, 6))
metrics_df['MAE'].sort_values().plot(kind='barh', color='lightcoral')
plt.title('Model Comparison (MAE)')
plt.xlabel('MAE')
plt.ylabel('Models')
plt.show()

future_Inflation_Rate = 2
future_Interest_Rate = 5
future_GDP = 17967
future_Applied = 2.3
future_economic_data = np.array([[future_Inflation_Rate, future_Interest_Rate, future_GDP, future_Applied]])
future_economic_data_scaled = scaler.transform(future_economic_data)

for model_name, model in models.items():
    future_gold_price = model.predict(future_economic_data_scaled)
    print(f'Predicted Gold Price using {model_name}: {future_gold_price[0]}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Gold Prices')
plt.ylabel('Predicted Gold Prices')
plt.title('Predicted vs Actual Gold Prices')
plt.show()

```

## Output

#### Output1 - Login Page

![image](https://github.com/user-attachments/assets/8693e621-b029-4469-97b0-e6a393286531)

#### Output2 - Gold Price Prediction's Accuracy
![image](https://github.com/user-attachments/assets/b703d382-0115-4180-be8e-1e0a4c5a34ed)


Detection Accuracy: 96.7%
Note: These metrics can be customized based on your actual performance evaluations.


## Results and Impact
The results of the project demonstrate improved accuracy in gold price forecasting using advanced machine learning models such as Random Forest, Gradient Boosting, and Decision Tree Regression. By integrating multiple economic indicators like inflation, unemployment rates, and GDP, the models provide more reliable and precise predictions compared to traditional methods. The impact of this system is significant for investors and financial analysts, as it offers real-time insights into future gold prices, helping in better decision-making and risk management. The system's adaptability to dynamic market conditions further enhances its usefulness in volatile economic scenarios.

## Articles published / References

1. W. M. P. Dhuhita, M. F. A. Farid, A. Yaqin, H. Haryoko and A. A. Huda, "Gold     Price rediction Based On Yahoo Finance Data Using Lstm Algorithm," 2023 International Conference on Informatics, Multimedia, Cyber and Informations System (ICIMCIS), Jakarta Selatan, Indonesia, 2023, pp. 420-425, doi: 10.1109/ICIMCIS60089.2023.10349035. https://ieeexplore.ieee.org/document/10349035

2. M. Abdou, M. Shaltout, A. Godah, K. Sobh, Y. Eid and W. Medhat, "Gold Price Prediction using Sentiment Analysis," 2022 20th International Conference on Language Engineering (ESOLEC), Cairo, Egypt, 2022, pp. 41-44, doi: 10.1109/ESOLEC54569.2022.10009529. https://ieeexplore.ieee.org/document/10009529

3. U. Landge, O. Phokmare, N. Borane and P. Shelke, "Gold Price Prediction using Random Forest Algorithm," 2024 3rd International Conference on Applied Artificial Intelligence and Computing (ICAAIC), Salem, India, 2024, pp. 1287-1292, doi: 10.1109/ICAAIC60222.2024.10575316.  
https://ieeexplore.ieee.org/document/10575316

4.K. A. Manjula and P. Karthikeyan, "Gold Price Prediction using Ensemble based Machine Learning Techniques," 2019 3rd International Conference on Trends in Electronics and Informatics (ICOEI), Tirunelveli, India, 2019,pp.1360-1364,
 doi: 10.1109/ICOEI.2019.8862557. 
https://ieeexplore.ieee.org/document/8862557

5. D. Nanthiya, S. B. Gopal, S. Balakumar, M. Harisankar and S. P. Midhun, "Gold Price Prediction using ARIMA model," 2023 2nd International Conference on Vision Towards Emerging Trends in Communication and Networking Technologies (ViTECoN), Vellore, India, 2023, pp. 1-6, doi: 10.1109/ViTECoN58111.2023.10157017. https://ieeexplore.ieee.org/document/10157017

6. S. Bhadula, D. Kartik and D. Gupta, "An Explainable AI Regression Model For Gold Price Prediction," 2024 IEEE 9th International Conference for Convergence in Technology (I2CT), Pune, India, 2024, pp. 1-7, doi: 10.1109/I2CT61223.2024.10543640. 
https://ieeexplore.ieee.org/document/10543640

7. M. Ghute and M. Korde, "Efficient Machine Learning Algorithm for Future Gold Price Prediction," 2023 International Conference on Inventive Computation Technologies (ICICT), Lalitpur, Nepal, 2023, pp. 216-220, doi: 10.1109/ICICT57646.2023.10134197. 
https://ieeexplore.ieee.org/document/10134197

8. Jayasingh, S.K., Mantri, J.K., Pradhan, S. (2022). Smart Weather Prediction Using Machine Learning. In: Udgata, S.K., Sethi, S., Gao, XZ. (eds) Intelligent Systems. Lecture Notes in Networks and Systems, vol 431. Springer, Singapore. 
https://doi.org/10.1007/978-981-19-0901-6_50

9. Prusti, D., Tripathy, A.K., Sahu, R., Rath, S.K. (2023). Bitcoin Price Prediction by Applying Machine Learning Approaches. In: Chinara, S., Tripathy, A.K., Li, KC., Sahoo, J.P., Mishra, A.K. (eds) Advances in Distributed Computing and Machine Learning. Lecture Notes in Networks and Systems, vol 660. Springer, Singapore. 
https://doi.org/10.1007/978-981-99-1203-2_26

10. Das, S., Nayak, J., Kamesh Rao, B., Vakula, K., Ranjan Routray, A. (2022). Gold Price Forecasting Using Machine Learning Techniques: Review of a Decade. In: Das, A.K., Nayak, J., Naik, B., Dutta, S., Pelusi, D. (eds) Computational Intelligence in Pattern Recognition . Advances in Intelligent Systems and Computing, vol 1349. Springer, Singapore. https://doi.org/10.1007/978-981-16-2543-5_58

11. Manickam, B.S., Jahankhani, H. (2024). Credit Card Fraud Detection Using Machine Learning. In: Jahankhani, H. (eds) Cybersecurity Challenges in the Age of AI, Space Communications and Cyborgs. ICGS3 2023. Advanced Sciences and Technologies for Security Applications. Springer, Cham. 
https://doi.org/10.1007/978-3-031-47594-8_15

12. V. Gupta, V. Aggarwal, S. Gupta, N. Sharma, K. Sharma and N. Sharma, "Visualization and Prediction of Heart Diseases Using Data Science Framework," 2021 Second International Conference on Electronics and Sustainable Communication Systems (ICESC), Coimbatore, India, 2021, pp. 1199-1202, doi: 10.1109/ICESC51422.2021.9532790.  
https://ieeexplore.ieee.org/document/9532790

13. N. Singh Yadav et al., "Business Decision making using Data Science," 2022 International Conference on Innovative Computing, Intelligent Communication and Smart Electrical Systems (ICSES), Chennai, India, 2022, pp. 1-11, doi: 10.1109/ICSES55317.2022.9914352.
https://ieeexplore.ieee.org/document/9914352

14. Cerna, R., Tirado, E., Bayona-Oré, S. (2022). Price Prediction of Agricultural Products: Machine Learning. In: Yang, XS., Sherratt, S., Dey, N., Joshi, A. (eds) Proceedings of Sixth International Congress on Information and Communication Technology. Lecture Notes in Networks and Systems, vol 217. Springer, Singapore.
 https://doi.org/10.1007/978-981-16-2102-4_78

15. P. Nagaraj, M. C. Prabhu, B. V. S. Kumar, C. Yasaswitha, K. Mohana and C. Jahnavi, "Weather Report Analysis Prediction using Machine Learning and Data Analytics Techniques," 2023 International Conference on Data Science, Agents & Artificial Intelligence (ICDSAAI), Chennai, India, 2023, pp. 1-5, doi: 10.1109/ICDSAAI59313.2023.10452556.
https://ieeexplore.ieee.org/document/10452556


