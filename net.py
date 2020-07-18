import numpy as np
import matplotlib.pyplot as plt
# 引用S函数
import scipy.special as sp


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
        self.actions_function = lambda x:sp.expit(x)
        pass



    # 训练神经网络
    def train(self,inputs_list,target_list):
      inputs = np.array(inputs_list,ndmin=2).T
      targets = np.array(target_list,ndmin=2).T

       #输入转置
      inputs=np.array(inputs_list,ndmin=2).T
      # 隐藏点输入
      hidden_inputs = np.dot(self.wih,inputs)
      # 隐藏节点输出
      hidden_outputs =self.actions_function(hidden_inputs)
      #最终节点输入
      final_inputs = np.dot(self.who,hidden_outputs)
      #隐藏点输出
      final_outputs = self.actions_function(final_inputs)
      #误差计算
      out_error = targets-final_outputs
      #隐藏节点误差输出（误差的反向传播）
      hidden_error = np.dot(self.who.T,out_error)

      #更新隐藏节点与输出节点的权重矩阵
      self.who += self.learningrate*np.dot((out_error*final_outputs*(1-final_outputs)),inputs.T)
      #更新输出节点与隐藏节点的权重矩阵
      self.wih += self.learningrate*np.dot((hidden_error*hidden_outputs*(1-hidden_outputs)),hidden_outputs.T)



      pass
    



    # 给定输入，输出
    def query(self,inputs_list):
      #输入转置
      inputs=np.array(inputs_list,ndmin=2).T
      # 隐藏点输入
      hidden_inputs = np.dot(self.wih,inputs)
      # 隐藏节点输出
      hidden_outputs =self.actions_function(hidden_inputs)
      #最终节点输入
      final_inputs = np.dot(self.who,hidden_outputs)
      #隐藏点输出
      final_outputs = self.actions_function(final_inputs)

      return final_outputs


      pass


    pass





inputnodes = 3
outputnodes = 3
hiddennodes = 3
learningrate = 0.5
# 创建神经网络
n=neuralNetwork(inputnodes, outputnodes, hiddennodes, learningrate)

final_outputs=n.query([1.0,0.5,-1.5])
print(final_outputs)