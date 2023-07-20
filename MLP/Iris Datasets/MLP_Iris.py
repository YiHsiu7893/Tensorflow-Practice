"""
brief intro.
運用MLP模型的實戰範例
載入鳶尾花資料集，並訓練一個能夠分辨鳶尾花種類的模型
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# 取得測試資料
iris = datasets.load_iris()  # 鳶尾花資料集
category = 3                 # 鳶尾花有幾種Label答案Y
dim = 4                      # 鳶尾花有幾個Feature特徵X
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
#print(x_train.shape)  # (120, 4)
#print(x_test.shape)   # (30, 4)
#print(y_train.shape)  # (120,)
#print(y_test.shape)   # (30,)

# Label預處理: One-Hot Encoding
y_train2 = tf.keras.utils.to_categorical(y_train, num_classes=category)
y_test2 = tf.keras.utils.to_categorical(y_test, num_classes=category)

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10, activation='relu', input_dim=dim))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=category, activation='softmax'))

# 編譯&訓練模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
model.fit(x_train, y_train2,
          epochs=50,
          batch_size=16)

# 評估正確率
score = model.evaluate(x_test, y_test2, batch_size=128)
print("accuracy=", score[1])

# 預測
predict = model.predict(x_test)
print("predict:", predict)
y_predict = np.argmax(predict, axis=1)
print("   Ans:", y_predict)
print("y_test:", y_test)
