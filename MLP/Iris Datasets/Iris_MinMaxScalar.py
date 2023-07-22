from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# 取得測試資料
iris = datasets.load_iris()
category = 3
dim = 4
# Feature預處理: 標準化
scalar = preprocessing.MinMaxScaler()             # 初始化函式
X_train_minmax = scalar.fit_transform(iris.data)  # 將資料轉換到0~1
# 資料分割
x_train, x_test, y_train, y_test = train_test_split(X_train_minmax, iris.target, test_size=0.2)
# Label預處理: One-Hot Encoding
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
                    epochs=100,
                    batch_size=16)

# 評估正確率
score = model.evaluate(x_test, y_test2, batch_size=128)
print("accuracy=", score[1])

# 顯示訓練過程
print(history.history.keys())          # 顯示紀錄的項目
plt.plot(history.history['accuracy'])  # 繪出訓練過程的正確率
plt.title('model acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['train acc'], loc='lower right')
plt.show()