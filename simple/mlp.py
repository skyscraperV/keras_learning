import numpy as np
from keras.callbacks import TensorBoard
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils.keras_save import export_savedmodel_pb

data_size = 200 # 生成的数据集大小
batch_size = 40
learning_rate = 0.0001
epochs = 50
model_version = "1"

model_path = "model/mlp.pb/"+ model_version
tfboard_log_apth = "model/mlp_tfboard_log/" + model_version

tbCallBack = TensorBoard(log_dir=tfboard_log_apth)


x = np.linspace(-10, 10, data_size)
# 使得数据点呈y=3x+5的关系，添加小范围正态随机数以获得数据的随机波动效果
y = 3 * x + 5 + np.random.rand(*x.shape)*0.3

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

# 创建一个Sequential模型
model = Sequential()

# 添加一个全连接层，输入数据维度为1，含一个输出单元，RandomUniform表示权值在正负0.05间均匀随机初始化
model.add(Dense(units=10, input_dim=1, kernel_initializer='RandomUniform'))
model.add(Dense(units=1))

# 打印查看当前的权重
print(model.layers[0].get_weights())

# 创建一个SGD对象以调整学习速率
sgd = optimizers.SGD(lr=learning_rate)
# 编译model，优化器使用刚创建的SGD对象，损失函数使用最小均方差mse
model.compile(optimizer=sgd, loss='mse')
# 使用之前生成的数据训练
model.fit(x_train, y_train, batch_size=batch_size ,epochs=epochs,callbacks=[tbCallBack])

# 再次打印权重，可以看到其值在3与5附近
print(model.layers[0].get_weights())

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=batch_size )

print(loss_and_metrics)

model.save("model/mlp.h5")

tf.keras.models.save_model(model, model_path)
