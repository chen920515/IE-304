import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D
import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from matplotlib import pyplot as plt
import matplotlib as mp

tf.random.set_seed(0)
mp.style.use('default')

# Load dataset
df = pd.read_csv('beer.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Transform dataset function
def Tranform_dataset(input_length, output_length, data):  
    df = data.copy()
    
    # Input columns creation
    for i in range(input_length):
        df[f'x_{i}'] = df['deg'].shift(-i)
    
    # Output columns creation
    for j in range(output_length):
        df[f'y_{j}'] = df['deg'].shift(-input_length-j)
    
    # Drop rows with NaN values
    df = df.dropna(axis=0)
    return df

history = 12  # Last values used by model
future = 6   # Predict future values

# Transform the dataset
full_data = Tranform_dataset(history, future, df)

# Split into input and output
X_cols = [col for col in full_data.columns if col.startswith('x')]
y_cols = [col for col in full_data.columns if col.startswith('y')]

X = full_data[X_cols].values
y = full_data[y_cols].values

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Reshape the data to fit the model
X_train = X_train.reshape(X_train.shape[0], history, 1)
X_test = X_test.reshape(X_test.shape[0], history, 1)

# Define the CNN-LSTM model
def get_model_cnn_lstm(history, future):
    model = Sequential()
    model.add(tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(LSTM(16, return_sequences=True, activation='relu'))
    model.add(LSTM(16, return_sequences=False, activation='relu'))
    model.add(Dense(future))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model_cnnlstm = get_model_cnn_lstm(history, future)
model_cnnlstm.summary()

# Train the model
filepath = 'CNN_LSTM.hdf5'
checkpoint_cnn_lstm = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss',
                             verbose=0, 
                             save_best_only=True,
                             mode='min')
callbacks_cnn_lstm = [checkpoint_cnn_lstm]

hist_cnn_lstm = model_cnnlstm.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2, validation_data=(X_test, y_test), 
                 shuffle=True, callbacks=callbacks_cnn_lstm)

# Plot training loss
fig = plt.figure(figsize=(13,5))
plt.rcParams.update({'font.size': 18})
plt.plot(hist_cnn_lstm.history['val_loss'], label="Loss CNN LSTM", lw=3)
plt.xlabel("Iterations", fontsize=20)
plt.ylabel("Loss", fontsize=20)
plt.legend(fontsize=18, loc='best', shadow=True)
plt.show()

# Get predictions on all the data
model_cnnlstm.load_weights('CNN_LSTM.hdf5')
y_pred_cnn_lstm = model_cnnlstm.predict(X.reshape(X.shape[0], history, 1))

pred_cnn_lstm = []
truth = []

for i in range(len(y_pred_cnn_lstm)):
    if i == (len(y_pred_cnn_lstm) - 1):
        for j in range(len(y_pred_cnn_lstm[i])):
            pred_cnn_lstm.append(y_pred_cnn_lstm[i][j])
            truth.append(y[i][j])    
    else:
        pred_cnn_lstm.append(y_pred_cnn_lstm[i][0])
        truth.append(y[i][0])

# Plot true values and predictions
years = df.index

# Generate months range corresponding to the length of the truth
months = np.arange(len(truth))

# Define the start points and labels for years on x-axis
year_starts = [i for i in range(0, len(truth), 84)]  # Adjust according to your data frequency
year_labels = [str(years[i].year) for i in year_starts]

# Plotting the true values and predictions
fig = plt.figure(figsize=(13, 5))
plt.rcParams.update({'font.size': 18})

# Plot the training data
plt.plot(months[0:len(y_train)], truth[0:len(y_train)], label='Train Data', lw=3)
plt.plot(months[len(y_train):], truth[len(y_train):], label='Test Data', lw=3)
plt.plot(pred_cnn_lstm, label='CNN_LSTM Predictions', lw=3, linestyle='dashed')

# Add vertical line to separate training and testing data
plt.vlines(x=len(y_train), ymin=50, ymax=230, lw=3, linestyle='dashed', color='r')

plt.xlabel('Year', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.xticks(year_starts, year_labels)
plt.legend(fontsize=18, loc='lower right', shadow=True)
plt.show()

# Calculate performance metrics
# Convert lists to numpy arrays
pred_cnn_lstm = np.array(pred_cnn_lstm)
truth = np.array(truth)

# Split the truth and predictions into training and testing parts
truth_train = truth[:len(y_train)]
truth_test = truth[len(y_train):]

pred_train = pred_cnn_lstm[:len(y_train)]
pred_test = pred_cnn_lstm[len(y_train):]

print(len(pred_train))
print(len(pred_test))

# Calculate metrics
mse_train = mean_squared_error(truth_train, pred_train)
mae_train = mean_absolute_error(truth_train, pred_train)
mape_train = mean_absolute_percentage_error(truth_train, pred_train)

mse_test = mean_squared_error(truth_test, pred_test)
mae_test = mean_absolute_error(truth_test, pred_test)
mape_test = mean_absolute_percentage_error(truth_test, pred_test)

print(f'Training MSE: {mse_train:.4f}')
print(f'Training MAE: {mae_train:.4f}')
print(f'Training MAPE: {mape_train:.4f}')

print(f'Testing MSE: {mse_test:.4f}')
print(f'Testing MAE: {mae_test:.4f}')
print(f'Testing MAPE: {mape_test:.4f}')

# Export predictions and true values to Excel
data = {
    'Truth': truth,
    'Predictions': pred_cnn_lstm
}

df_output = pd.DataFrame(data)

# Export to Excel
df_output.to_excel('truth_and_predictions.xlsx', index=False)

print("Data has been successfully exported to 'truth_and_predictions.xlsx'")