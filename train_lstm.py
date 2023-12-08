import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Đọc dữ liệu
file_paths = ["DATASET\dau_sac.txt",
              "DATASET\dau_huyen.txt",
              "DATASET\chanh_muoi.txt",
              "DATASET\chanh_da.txt",
              "DATASET\\xin_chao.txt", 
              "DATASET\\toi_yeu_ban.txt",
              "DATASET\\aw.txt",
              "DATASET\\aa.txt",
              "DATASET\ee.txt",
              "DATASET\oo.txt",
              "DATASET\\uw.txt"]

no_of_timesteps = 30
X = []
y = []

for i, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)
    dataset = df.iloc[:, 1:].values
    n_sample = len(dataset)
    for j in range(no_of_timesteps, n_sample):
        X.append(dataset[j - no_of_timesteps:j, :])
        y.append(i)

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

# Cấu trúc mô hình
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=11, activation="softmax"))
model.compile(optimizer="adam", metrics=['accuracy'], loss="categorical_crossentropy")

y = to_categorical(y, num_classes=11)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #TRAIN 80% VÀ TEST 20%

model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
model.save("model.h5")
