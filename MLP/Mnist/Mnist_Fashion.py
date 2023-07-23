"""
brief intro.
運用MLP模型的實戰範例
載入Fashion_Mnist服飾資料集，並訓練一個能夠分辨服飾種類的模型
可用編號或是文字列印標籤答案
*與Mnist.py作法幾乎相同
"""
import tensorflow as tf
import numpy as np

# 載入服飾資料
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
#print("x_train = " + str(x_train.shape))  # x_train = (60000, 28, 28)
#print("y_train = " + str(y_train.shape))  # y_train = (60000,)
#print("x_test = " + str(x_test.shape))    # x_test = (10000, 28, 28)
#print("y_test = " + str(y_test.shape))    # y_test = (10000,)
category = 10
dim = 28*28

# Feature預處理
# 資料轉換為1D
x_train = x_train.reshape(x_train.shape[0], dim)
x_test = x_test.reshape(x_test.shape[0], dim)
# 將原數據標準化至0~1的浮點數
x_train = x_train.astype('float32')
x_test = x_test.astype('float')
x_train /= 255
x_test /= 255

# Label預處理
y_train2 = tf.keras.utils.to_categorical(y_train, num_classes=category)
y_test2 = tf.keras.utils.to_categorical(y_test, num_classes=category)

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10, activation='relu', input_dim=dim))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=category, activation='softmax'))

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
# 以編號列印標籤答案
print("前十項預測答案:", y_predict[:10])
print("前十項正確答案:", y_test[:10])
# 以文字列印標籤答案
clothes = ["T恤", "褲子", "套頭衫", "禮服", "外套", "涼鞋", "襯衫", "運動鞋", "袋子", "長靴"]
str1 = str2 = ""
for i in range(0, 10):
    str1 = str1 + ("%3s " % clothes[int(y_predict[i])])
    str2 = str2 + ("%3s " % clothes[int(y_test[i])])
print("前十項預測答案: [ %s]" % str1)
print("前十項正確答案: [ %s]" % str2)