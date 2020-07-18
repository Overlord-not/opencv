import numpy as np
import matplotlib.pyplot as plt
# 引用S函数
import scipy.special as sp
import matplotlib.pyplot as plt


class neuralNetwork(object):
    # 初始化神经网络
    def __init__(self, inputnodes, outputnodes, hiddennodes, learningrate):
        self.inputnodes = inputnodes
        self.outputnodes = outputnodes
        self.hiddennodes = hiddennodes
        self.learningrate = learningrate
        # 创建初始权重矩阵
        self.wih = np.random.rand(self.hiddennodes, self.inputnodes)-0.5
        self.who = np.random.rand(self.outputnodes, self.hiddennodes)-0.5
        # 以正太分布的方式创建初始权重矩阵
        # self.wih=np.random.normal(0.0,power(self.hiddennodes,-0.5),(self.hiddennodes,self.inputnodes))
        # self.who=np.random.normal(0.0,power(self.outputnodes,-0.5),(self.outputnodes,self.hiddennodes))
        # 神经网络抑制信号函数(具体可Google lambda函数的用法)
        self.actions_function = lambda x: sp.expit(x)
        pass

    # 训练神经网络
    def train(self, inputs_list, target_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # 输入转置
        inputs = np.array(inputs_list, ndmin=2).T
        # 隐藏点输入
        hidden_inputs = np.dot(self.wih, inputs)
        # 隐藏节点输出
        hidden_outputs = self.actions_function(hidden_inputs)
        # 最终节点输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 隐藏点输出
        final_outputs = self.actions_function(final_inputs)
        # 误差计算
        out_error = targets-final_outputs
        # 隐藏节点误差输出（误差的反向传播）
        hidden_error = np.dot(self.who.T, out_error)

        # 更新隐藏节点与输出节点的权重矩阵
        self.who += self.learningrate * \
            np.dot(out_error*final_outputs*(1-final_outputs), hidden_inputs.T)
        # 更新输出节点与隐藏节点的权重矩阵
        self.wih += self.learningrate * \
            np.dot(hidden_error*hidden_outputs*(1-hidden_outputs), inputs.T)

        pass

    # 给定输入，输出

    def query(self, inputs_list):
        # 输入转置
        inputs = np.array(inputs_list, ndmin=2).T
        # 隐藏点输入
        hidden_inputs = np.dot(self.wih, inputs)
        # 隐藏节点输出
        hidden_outputs = self.actions_function(hidden_inputs)
        # 最终节点输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 隐藏点输出
        final_outputs = self.actions_function(final_inputs)

        return final_outputs

        pass

    pass


inputnodes = 784
outputnodes = 10
hiddennodes = 100
learningrate = 0.3
# 创建神经网络
n = neuralNetwork(inputnodes, outputnodes, hiddennodes, learningrate)

# 读取训练集
data_file = open('mnist_train_100.csv', 'r')
data_list = data_file.readlines()
data_file.close()

for i in range(len(data_list)):
    # 以逗号把读取的一个长字符串进行拆分一个个字符串并创建含字符串的列表
    all_values = data_list[i-1].split(',')
    # 忽略第一个数字进行矩阵转换（28*28），函数asfrry字符串转换为数字并创建列表
    inputs = (np.asfarray(all_values[1:])/255*0.99+0.01)
    # 绘制列表
    # plt.imshow(image_array,cmap='gray',interpolation='none')
    # plt.show()
    # 创建目标值 10个输出
    targets = np.zeros(outputnodes)+0.01
    # 读取训练集的第一个标签值，这是需要的目标值
    targets[int(all_values[0])] = 0.99
    #训练100次
    n.train(inputs, targets)
    pass




data_file = open('mnist_test_10.csv', 'r')
data_list = data_file.readlines()
data_file.close()

# 以逗号把读取的一个长字符串进行拆分一个个字符串并创建含字符串的列表
all_values = data_list[0].split(',')
# 忽略第一个数字进行矩阵转换（28*28），函数asfrry字符串转换为数字并创建列表
inputs = (np.asfarray(all_values[1:])/255.0*0.99+0.01)

final_outputs = n.query(inputs)
print(final_outputs)
