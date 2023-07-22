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
scalar = preprocessing.MinMaxScaler()
X_train_minmax = scalar.fit_transform(iris.data)
# 資料分割
x_train, x_test, y_train, y_test = train_test_split(X_train_minmax, iris.target, test_size=0.2)
# Label預處理: One-Hot Encoding
y_train2 = tf.keras.utils.to_categorical(y_train, num_classes=category)
y_test2 = tf.keras.utils.to_categorical(y_test, num_classes=category)

# 函式model()，建立模型並以learning rate=lr進行訓練
def model(lr):
    # 建立模型
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=10, activation='relu', input_dim=dim))
    model.add(tf.keras.layers.Dense(units=10, activation='relu'))
    model.add(tf.keras.layers.Dense(units=category, activation='softmax'))

    # 訓練模型
    opt1 = tf.keras.optimizers.Adam(learning_rate=lr)  # 調整學習率
    model.compile(optimizer=opt1,
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train2,
                        epochs=100,
                        batch_size=16)
    
    return history                                     # 回傳訓練紀錄

# 以不同learning rate訓練模型，並取得訓練紀錄
history_01 = model(0.01)                               # 學習率0.01
history_001 = model(0.001)                             # 學習率0.001

# 觀察各學習率的訓練過程
plt.plot(history_01.history['accuracy'])               # 顯示學習率0.01訓練時的正確率
plt.plot(history_001.history['accuracy'])              # 顯示學習率0.001訓練時的正確率
plt.title('Adam model acc rate')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['rate 0.01', 'rate 0.001'], loc='lower right')
plt.show()