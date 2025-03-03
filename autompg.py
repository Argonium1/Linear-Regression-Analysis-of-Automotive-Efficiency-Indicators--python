import pathlib
import numpy as np #numpy==1.18
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf #tensorflow==2.1.0
from tensorflow import keras
from tensorflow.keras import layers

dataset_path = pd.read_csv('F:\\Desktop\\test\\auto-mpg.csv')

#数据处理
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
              'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()  # 复制读取的数据,避免修改元数据

# dataset.head()
# dataset.isna().sum()  # 显示无效数据分布情况

dataset = dataset.dropna()  # 移除无效数据

# dataset.isna().sum()  # 显示无效数据分布情况

origin = dataset.pop('Origin')  # 获取分类序号——以供one-hot转换需要

# origin

dataset['USA'] = (origin == 1)*1.0
# one-hot转换:只有当列数据中值为1时,才设为1,否则为0,也就相当于一整列数据按 (origin == 1)条件转换为1 or 0

dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0

# dataset.head()

#训练集和测试集
train_dataset = dataset.sample(frac=0.8,random_state=0)  # 拆分获得训练数据
test_dataset = dataset.drop(train_dataset.index)  # 拆分获得测试数据

train_stats = train_dataset.describe()  # 获取数据的一系列描述信息
train_stats.pop("MPG")  # 移除MPG的数据列
train_stats = train_stats.transpose()  # 行列转换——也就是相当于翻转

#分离训练数据的标签
train_labels = train_dataset.pop('MPG')   # 将移除的MPG列数据返回,赋值给train_labels
test_labels = test_dataset.pop('MPG')
train_labels   # 显示数据

# train_labels
# test_labels

#数据标准化

def norm(x):
    return (x train_stats['mean']) / train_stats['std']

    normed_train_data = norm(train_dataset)  # 获取训练数据中的归一化数据
    normed_test_data = norm(test_dataset)  # 获取测试归一化数据

    #模型构建
    # 第一层Dense的input_shape为输入层大小,前边的64为该全连接层的输出层(隐藏层)——只有最后的全连接层 layers.Dense(1)输出才是输出层,否则为隐藏层.
    # activation为激活函数——这里是线性激活
def build_model():
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
       optimizer=optimizer,
       metrics=['mae', 'mse'])
    return model


#模型检测
example_batch = normed_train_data[:10]  # 获取十个数据来预测
example_result = model.predict(example_batch)

#example_result

# 通过为每个完成的时期打印一个点来显示训练进度
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

        EPOCHS = 1000  # 拟合次数

        # 返回的history为一个对象,内包含loss信息等
        history = model.fit(
        normed_train_data, train_labels,
        epochs=EPOCHS, validation_split = 0.2, verbose=0,callbacks=[PrintDot()])

        hist = pd.DataFrame(history.history)  # 返回一个DataFrame对象,包含history的history数据
        hist['epoch'] = history.epoch  # 末尾添加一列数据包含epoch信息

        # hist.tail()  # 展示表单数据

        # 训练过程可视化
    def plot_history(history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

          plt.figure()  # 创建新窗口
          plt.xlabel('Epoch')  # x轴标签(名字)
          plt.ylabel('Mean Abs Error [MPG]')  # y轴标签(名字)
          plt.plot(hist['epoch'], hist['mae'],
                   label='Train Error')  # 绘图,label为图例
          plt.plot(hist['epoch'], hist['val_mae'],
                   label = 'Val Error')  # 绘图,label为图例
          plt.ylim([0,5])  # y轴长度限制
          plt.legend()  # 展示图例

          plt.figure()
          plt.xlabel('Epoch')
          plt.ylabel('Mean Square Error [$MPG^2$]')
          plt.plot(hist['epoch'], hist['mse'],
                   label='Train Error')
          plt.plot(hist['epoch'], hist['val_mse'],
                   label = 'Val Error')
          plt.ylim([0,20])
          plt.legend()
          plt.show()  # 必须要有show才能显示绘图

        # plot_history(history)

        # 模型改进

        model = build_model()  # 重新定义模型

        # patience 值用来检查改进 epochs 的数量
        # 定义callbacks的操作设置,这里采用了每10次fit,进行一次判断是否停下,判断依据是当val_loss不改变甚至降低.
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                            validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])



        loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)  # 返回第一个数据loss为损失值
        print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))  # mae为我们构建网络归回预测的值——即预测的MPG

        # plot_history(history)

        # 模型评估
        loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)  # 返回第一个数据loss为损失值
        print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))  # mae为我们构建网络归回预测的值——即预测的MPG

        # 训练数据可视化

        test_predictions = model.predict(normed_test_data).flatten()
        # 预测信息会被flatten()平铺展开后以一维数据返回

        plt.scatter(test_labels, test_predictions)  # 绘制散点图
        plt.xlabel('True Values [MPG]')
        plt.ylabel('Predictions [MPG]')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0,plt.xlim()[1]])
        plt.ylim([0,plt.ylim()[1]])
        _ = plt.plot([-100, 100], [-100, 100])  # 画一条y = x的直线,方便分析

        #误差分布

        error = test_predictions test_labels  # test_labels 是原始的MPG值序列,所以相减得到误差.
        plt.hist(error, bins = 25)  # 画矩形图——bins表示每次绘图的最大矩形数目
        plt.xlabel("Prediction Error [MPG]")
        _ = plt.ylabel("Count")
