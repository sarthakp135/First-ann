# First-ann

mport tensorflow
from tensorflow import keras
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

df = read_csv("C:/Users/Sarthak/Desktop/friction.csv")
X = df.drop('fatiguec', axis = 1)
X = X.drop('fatiguet', axis = 1)
y=df.drop('W', axis = 1)
y=y.drop('v', axis = 1)
y=y.drop('fatiguet', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
#X_train_scaled=X_train
X_test_scaled = scaler.transform(X_test)
#X_test_scaled=X_test
model = Sequential()
model.add(Dense(12, input_dim=2, activation='relu'))
#model.add(Dense(300, activation='softmax'))
#Output layer
model.add(Dense(1, activation="softmax"))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
#model.compile(loss='crossentropy',optimizer='sgd',metrics=['accuracy'])
model.summary()
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs =1000)


from matplotlib import pyplot as plt
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['mae']
val_acc = history.history['val_mae']
plt.plot(epochs, acc, 'y', label='Training MAE')
plt.plot(epochs, val_acc, 'r', label='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
model.evaluate(X_test,y_test)


############################################
#Predict on test data
predictions = model.predict(X_test_scaled)
print("Predicted values are: ", predictions)
print("Real values are: ", y_test)
##############################################

#Comparison with other models..
#Neural network - from the current code
mse_neural, mae_neural = model.evaluate(X_test_scaled, y_test)
print('Mean squared error from neural net: ', mse_neural)
print('Mean absolute error from neural net: ', mae_neural)
