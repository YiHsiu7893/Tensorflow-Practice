"""
brief intro.
運用CNN模型的實戰範例
載入Fashion_Mnist服飾資料集，並訓練一個能夠分辨服飾種類的模型
訓練結合儲存模型權重的功能: 利用train_on_batch每隔20次儲存權重
程式執行後會產生 fashion_mnist.json 和 fashion_mnist.h5 兩個檔案
"""
import tensorflow as tf
import numpy as np

# 載入服飾資料
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train[:2000]  # 減少數據
y_train = y_train[:2000]  # 減少數據
x_test = x_test[:200]     # 減少數據
y_test = y_test[:200]     # 減少數據
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

# Label預處理
y_train2 = tf.keras.utils.to_categorical(y_train, num_classes=category)
y_test2 = tf.keras.utils.to_categorical(y_test, num_classes=category)

# 建立CNN模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=28,    # 28個卷積濾鏡
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu',
                                 input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Conv2D(filters=56,    # 56個卷積濾鏡
                                 kernel_size=(2, 2),
                                 padding='same',
                                 activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.01))   # 丟掉1%的圖
model.add(tf.keras.layers.Flatten())            # 將2D圖像轉為1D
# MLP
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=category, activation='softmax'))

# 讀取模型權重
try:
    with open("fashion_mnist.h5", "r") as load_weights:
        model.load_weights("fashion_mnist.h5")  # 讀取模型權重繼續計算
except IOError:
    print("File not exists")

# 編譯
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# 訓練: train_on_batch，並每隔20次儲存權重
for step in range(1001):
    cost = model.train_on_batch(x_train, y_train2)
    print("step{}   train cost{}".format(step, cost))
    if step%20 == 0:                            # 每隔20次保存模型架構
        with open("fashion_mnist.json", "w") as json_file:
            json_file.write(model.to_json())    # 保存模型架構
        model.save_weights("fashion_mnist.h5")  # 保存模型權重

# 評估正確率
score = model.evaluate(x_test, y_test2)
print("score:", score)

# 預測
predict = np.argmax(model.predict(x_test), axis=1)
print("前十項預測答案:", predict[:10])
print("前十項正確答案:", y_test[:10])
