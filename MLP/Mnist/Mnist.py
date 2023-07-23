"""
brief intro.
運用MLP模型的實戰範例
載入Mnist手寫數字資料集，並訓練一個能夠分辨數字0~9的模型
"""
import tensorflow as tf
import numpy as np

# 取得手寫數字資料
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#print("x_train = " + str(x_train.shape))  # x_train = (60000, 28, 28)
#print("y_train = " + str(y_train.shape))  # y_train = (60000,)
category = 10  # 數字0~9
dim = 28*28    # 原始檔案為28*28的圖檔

# Feature預處理
# 資料轉換為1D
x_train = x_train.reshape(x_train.shape[0], dim)
x_test = x_test.reshape(x_test.shape[0], dim)
# 將原數據標準化至0~1的浮點數
x_train = x_train.astype('float32')  # 轉換至浮點數
x_test = x_test.astype('float32')
x_train /= 255                       # 原色彩數據為0~255，故除以255
x_test /= 255

# Label預處理
y_train2 = tf.keras.utils.to_categorical(y_train, num_classes=category)
y_test2 = tf.keras.utils.to_categorical(y_test, num_classes=category)

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10, activation='relu', input_dim=dim))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
kernel_initializer = 'normal'
model.add(tf.keras.layers.Dense(units=category, activation='softmax'))
#model.summary()  # 查看模型架構

# 訓練模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
model.fit(x_train, y_train2,
          epochs=200,
          batch_size=1000,
          verbose=1)

# 評估正確率
score = model.evaluate(x_test, y_test2, batch_size=128)
print("score:", score)

# 預測
predict = model.predict(x_test)
y_predict = np.argmax(predict, axis=1)
print("前十項預測答案:", y_predict[:10])
print("前十項正確答案:", y_test[:10])