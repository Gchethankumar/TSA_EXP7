### Developed By : G Chethan Kumar
### Register No. : 212222240022
### Date : 


# Ex.No: 07                                       AUTO REGRESSIVE MODEL

### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```python

import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('india-gdp.csv', parse_dates=['date'], index_col='date')

# Assuming 'AnnualChange' is the column with temperature data
ANNUALCHAnge_data = data['AnnualChange']

# Plot the temperature data
plt.figure(figsize=(10, 6))
plt.plot(AnnualChange_data.index, temperature_data, label='India-GDP Annual Change')  # Plot index vs. values
plt.title('Annual GDP Change Data')
plt.xlabel('Date')
plt.ylabel('Annual GDP change')
plt.legend()
plt.show()

# Perform ADF test
adf_result = adfuller(AnnualChange_data)
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')

# Split data into train and test sets
train_size = int(len(temperature_data) * 0.8)  # Use temperature_data for length
train, test = AnnualChange_data.iloc[:train_size], AnnualChange_data.iloc[train_size:]

# Adjust lag order to be less than the number of available data points
# Reduced lag order to 5 (you might need to experiment with this value)
lag_order = 5  

# Fit AutoReg model with the adjusted lag order
model = AutoReg(train, lags=lag_order)  
model_fitted = model.fit()


# Plot ACF and PACF with adjusted lags
plt.figure(figsize=(12, 8))
plt.subplot(211)
plot_acf(AnnualChange_data, lags=len(AnnualChange_data) // 2 -1, ax=plt.gca())  # Use AnnualChange_data for ACF, lags adjusted to be less than or equal to 50% of data length
plt.title('Autocorrelation Function (ACF)')

plt.subplot(212)
plot_pacf(AnnualChange_data, lags=len(AnnualChange_data) // 2 - 1, ax=plt.gca())  # Use AnnualChange_data for PACF, lags adjusted to be less than or equal to 50% of data length
plt.title('Partial Autocorrelation Function (PACF)')


plt.tight_layout()
plt.show()

# Make predictions
predictions = model_fitted.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Plot predictions vs. actual test data
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual Test Data', color='blue')
plt.plot(test.index, predictions, label='Predicted Data', color='red')
plt.title('Test Data vs Predictions')
plt.xlabel('Date')
plt.ylabel('Annual Change in GDP') 
plt.legend()
plt.show()

# Calculate and print MSE
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')



# Make final prediction
final_prediction = model_fitted.predict(start=len(temperature_data), end=len(temperature_data))
print(f'Final Prediction for Next Time Step: {final_prediction[len(AnnualChange_data)]}')

```
### OUTPUT:

GIVEN DATA

![Screenshot 2024-10-18 111409](https://github.com/user-attachments/assets/1519ae9f-6148-49aa-a086-2c0d264ab5f1)

PACF - ACF

![Screenshot 2024-10-18 111428](https://github.com/user-attachments/assets/a585029b-10ea-40d1-9ba5-54b6a0d9f5b9)


PREDICTION

![Screenshot 2024-10-18 112252](https://github.com/user-attachments/assets/b90228aa-7fe9-4b45-bb81-c08a73c6f98b)

FINIAL PREDICTION

![Screenshot 2024-10-18 112300](https://github.com/user-attachments/assets/3d9eaa4e-6b09-4c3d-8786-96990177e240)

### RESULT:
Thus we have successfully implemented the auto regression function using python.
