from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# 取得資料
iris = datasets.load_iris()
category = 3
dim = 4
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
y_train2 = tf.keras.utils.to_categorical(y_train, num_classes=category)
y_test2 = tf.keras.utils.to_categorical(y_test, num_classes=category)

# 函式model()，建立模型並以optimizer=opt1進行訓練
def model(opt1):
    # 建立模型
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=10, activation='relu', input_dim=dim))
    model.add(tf.keras.layers.Dense(units=10, activation='relu'))
    model.add(tf.keras.layers.Dense(units=category, activation='softmax'))

    # 訓練模型
    model.compile(optimizer=opt1,  # 指定optimizer為opt1
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train2,
                        epochs=80,
                        batch_size=16)
    
    return history                 # 回傳訓練紀錄

# 以不同optimizer訓練模型，並取得訓練紀錄
lr = 0.01                          # 學習率
history_Adam = model(tf.keras.optimizers.Adam(learning_rate=lr))              # 最佳化Adam
history_SGD = model(tf.keras.optimizers.SGD(learning_rate=lr))                # 最佳化SGD
history_RMSprop = model(tf.keras.optimizers.RMSprop(learning_rate=lr))        # 最佳化RMSprop
history_Adagrad = model(tf.keras.optimizers.Adagrad(learning_rate=lr))        # 最佳化Adagrad
history_Adadelta = model(tf.keras.optimizers.Adadelta(learning_rate=lr))      # 最佳化Adadelta
history_Nadam = model(tf.keras.optimizers.Nadam(learning_rate=lr))            # 最佳化Nadam
history_mom = model(tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9))  # 最佳化Momemtum

# 觀察各最佳化方式的訓練過程
plt.plot(history_Adam.history['accuracy'])                                    # 顯示Adam訓練時的正確率
plt.plot(history_SGD.history['accuracy'])                                     # 顯示SGD訓練時的正確率
plt.plot(history_RMSprop.history['accuracy'])                                 # 顯示RMSprop訓練時的正確率
plt.plot(history_Adagrad.history['accuracy'])                                 # 顯示Adagrad訓練時的正確率
plt.plot(history_Adadelta.history['accuracy'])                                # 顯示Adadelta訓練時的正確率
plt.plot(history_Nadam.history['accuracy'])                                   # 顯示Nadam訓練時的正確率
plt.plot(history_mom.history['accuracy'])                                     # 顯示Momentum訓練時的正確率
plt.title('optimizers acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Nadam', 'Momentum'], loc='lower right')
plt.show()