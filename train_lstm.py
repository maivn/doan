import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split

# Đọc dữ liệu
statica_df = pd.read_csv("STOP.txt")
staticb_df = pd.read_csv("THIS MARSHALLER.txt")
staticc_df = pd.read_csv("PROCEED TO NEXT MARSHALLER ON THE RIGHT.txt")
staticd_df = pd.read_csv("PROCEED TO NEXT MARSHALLER ON THE LEFT.txt")
statice_df = pd.read_csv("PERSONNEL APPROACH AIRCRAFT ON THE RIGHT.txt")
staticf_df = pd.read_csv("PERSONNEL APPROACH AIRCRAFT ON THE LEFT.txt")
normal_df = pd.read_csv("NORMAL.txt")
motiona_df = pd.read_csv("TURN TO THE LEFT.txt")
motionb_df = pd.read_csv("TURN TO THE RIGHT.txt")
motionc_df = pd.read_csv("SLOW DOWN.txt")
motiond_df = pd.read_csv("MOVE FORWARD.txt")

X = []
y = []
no_of_timesteps = 20

dataset = statica_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([1,0,0,0,0,0,0,0,0,0,0])

dataset = staticb_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,1,0,0,0,0,0,0,0,0,0])

dataset = staticc_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,1,0,0,0,0,0,0,0,0])

dataset = staticd_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,0,1,0,0,0,0,0,0,0])

dataset = statice_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,0,0,1,0,0,0,0,0,0])

dataset = staticf_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,0,0,0,1,0,0,0,0,0])

dataset = normal_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,0,0,0,0,1,0,0,0,0])

dataset = motiona_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,0,0,0,0,0,1,0,0,0])

dataset = motionb_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,0,0,0,0,0,0,1,0,0])

dataset = motionc_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,0,0,0,0,0,0,0,1,0])

dataset = motiond_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,0,0,0,0,0,0,0,0,1])

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model  = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(units = 11, activation="softmax"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "categorical_crossentropy")

model.fit(X_train, y_train, epochs=13, batch_size=32,validation_split=0.2)
model.save("model20.h5")


