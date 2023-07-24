"""
brief intro.
運用CNN模型的實戰範例
載入Mnist手寫數字資料集，並訓練一個能夠分辨數字0~9的模型
*提升正確率的方法:
1.增加Conv2d層
2.增加Conv2D層的filters數量
3.適量增減MaxPool2D、Dropout層的數量
"""
import tensorflow as tf
import numpy as np

# 載入手寫數字資料
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
category = 10

# Feature預處理
# 改變維度:(數量, 寬, 高, 顏色)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# 標準化
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
#print("x_train shape:", x_train.shape)    # x_train shape: (60000, 28, 28, 1)
#print(x_train.shape[0], "train samples")  # 60000 train samples
#print(x_test.shape[0], "test samples")    # 10000 test samples

# Label預處理
y_train2 = tf.keras.utils.to_categorical(y_train, num_classes=category)
y_test2 = tf.keras.utils.to_categorical(y_test, num_classes=category)

# 建立CNN模型
model = tf.keras.models.Sequential()
# 第一層由28*28*1處理後為28*28*3
model.add(tf.keras.layers.Conv2D(filters=3,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu',
                                 input_shape=(28, 28, 1)))
# 第二層處理後為14*14*3
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))  # 寬和高縮小一半
# 第三層處理後為14*14*9
model.add(tf.keras.layers.Conv2D(filters=9,
                                 kernel_size=(2, 2),
                                 padding='same',
                                 activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.5))            # 丟掉50%的圖
# 第五層處理後為14*14*6
model.add(tf.keras.layers.Conv2D(filters=6,
                                 kernel_size=(2, 2),
                                 padding='same',
                                 activation='relu'))
model.add(tf.keras.layers.Flatten())                    # 將2D圖像轉為1D
# MLP
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=category, activation='softmax'))
#model.summary()

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
predict = np.argmax(model.predict(x_test), axis=1)
print("前十項預測答案:", predict[:10])
print("前十項正確答案:", y_test[:10])