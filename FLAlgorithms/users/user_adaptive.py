import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import MySGD, FEDLOptimizer
from FLAlgorithms.users.userbase import User

# Implementation for Per-FedAvg clients

class UserPerAvg(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate,beta,lamda,
                 local_epochs, optimizer, total_users , num_users):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs)
        self.total_users = total_users
        self.num_users = num_users
        
        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate)

    def set_grads(self, new_grads):
        # 得保存一轮的模型结果
        self.prior_model = copy.deepcopy(list(self.model.parameters()))
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        self.model.train()
    def train(self, epochs):
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs):  # local update 
            self.model.train()

            temp_model = copy.deepcopy(list(self.model.parameters()))

            #step 1
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

            #step 2
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()

            # restore the model parameters to the one before first update
            for old_p, new_p in zip(self.model.parameters(), temp_model):
                old_p.data = new_p.data.clone()
            self.optimizer.step(beta = self.beta)

            # clone model to user model 
            self.clone_model_paramenter(self.model.parameters(), self.local_model)

        return LOSS

    def train_one_step(self):
        self.model.train()
        #step 1
        X, y = self.get_next_test_batch()
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step()
            #step 2
        X, y = self.get_next_test_batch()
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step(beta=self.beta)


# Implementation for Per-FedAvg clients
class UserAdaptive(UserPerAvg):
    # 类似于set_grads,一会单独写一个set_tao,server来制定tao
    def __init__(self, device, numeric_id, train_data, test_data, model,batch_size,  learning_rate, beta, lamda, 
    local_epochs, optimizer, total_users, num_users):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs)
        self.total_users = total_users
        self.num_users = num_users

        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate)
        pass

    def set_grads(self, new_grads):
        # 得保存一轮的模型结果
        self.prior_model = copy.deepcopy(list(self.model.parameters()))
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]
    def set_taos(self, new_taos):
        # new_taos是一个list，每一项对应一个新的tao
        pass

    def calculate_Fw(self):
        # 只取一个batch的数据来进行估计
        # 估计每一个用户的lou_hat 和 beta_hat
        self.model.train()

        temp_model = copy.deepcopy(list(self.model.parameters()))
        self.clone_model_paramenter(self.model.parameters(), self.prior_model)# 模型换成上一轮的
        X, y = self.get_next_train_batch()
        #step 1: 求t时本地模型的loss,向量求和取平均得到
        self.optimizer.zero_grad()
        tloc_output = self.model(X)

        tloc_loss = self.loss(tloc_output, y)
        tloc_grd = tloc_loss.backward()# 求loss一阶梯度
        # step 2: 求t时全局模型的loss,向量求和取平均
        self.clone_model_paramenter(self.model.parameters(), temp_model)# 模型换成新的
        self.optimizer.zero_grad()
        output = self.model(X)
        tglo_loss = self.loss(output, y)
        tglo_grd = tglo_loss.backward()# 求loss一阶梯度

        # 1、2得到的梯度和loss相减求范数再除以参数差的范数
        p_hat_i = torch.norm(tloc_loss-tglo_loss)/torch.norm(self.prior_model-temp_model)
        beta_hat_i = torch.norm(tloc_grd - tglo_grd)/torch.norm(self.prior_model-temp_model)

        return p_hat_i, beta_hat_i, tloc_loss, tloc_grd