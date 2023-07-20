from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 取得資料
iris = datasets.load_iris()
category = 3
dim = 4
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
y_train2 = tf.keras.utils.to_categorical(y_train, num_classes=category)
y_test2 = tf.keras.utils.to_categorical(y_test, num_classes=category)

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10, activation='relu', input_dim=dim))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=category, activation='softmax'))

# 訓練模型並記錄訓練過程
model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
history = model.fit(x_train, y_train2,
                    epochs=50,
                    batch_size=16)

# 評估正確率
#score = model.evaluate(x_test, y_test2, batch_size=128)
#print("accuracy=", score[1])

# 觀察訓練過程
plt.plot(history.history['accuracy'])  # 顯示訓練時的正確率accuracy
plt.plot(history.history['loss'])      # 顯示訓練時的損失率loss
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('acc & loss')
plt.legend(['acc', 'loss'], loc='upper left')
plt.show()