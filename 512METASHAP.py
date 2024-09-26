# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

import pathlib

from collections import OrderedDict, abc as container_abcs

from sklearn.model_selection import KFold
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from models import ResNet1D
from tqdm import tqdm
import shap
import matplotlib.pyplot as plt

# 读取CSV文件
data_path = '20240907PTCPDTCATC1.csv' #'data.csv'
f = open(data_path,"rb")# 二进制格式读文件
i = 0
while True:
    i += 1
    line = f.readline() # 按行读取

    if not line:
        break
    else:
        try:
            line.decode('ANSI')
        except: # 打印出不能通过'ANSI'方式解码的数据行
            print(i)
            print(str(line))

df = pd.read_csv(data_path, encoding='ANSI')
#20240907PTCPDTCATC.csv
# 提取样本ID、标签和特征
sample_ids = df.iloc[:, 0].values
labels_input = df.iloc[:, 1].values
features = df.iloc[:, 3:].values
features = features.astype(float) 
N_samples = len(sample_ids)
label_set = set(labels_input)
print(label_set)
num_class = len(label_set)
str_to_num = dict((c,i) for i,c in enumerate(label_set))
label_nums = torch.tensor([str_to_num[str_f] for str_f in labels_input])
labels = F.one_hot(label_nums)

# norm feature
print(features.shape)
features = torch.tensor(features)
'''
feat_mean = features.mean(dim=1,keepdim=True)
feat_std = features.std(dim=1,keepdim=True) + 1e-6
features = (features-feat_mean)/feat_std
feat_max, _ = features.max(dim=0)
feat_min, _ = features.min(dim=0) 
features = (features-feat_min)/(feat_max + 1e-6)
'''




#print(labels)
print(num_class)





# 超参数
batch_size = 32
num_epochs = 50
test_ep = 10
lr_init = 0.001
criterion = torch.nn.CrossEntropyLoss()
acc_list = []
acc_sum = 0.
n_splits = 5

# 定义K折交叉验证
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 数据集分割函数
def get_data_loader(X_train, y_train, X_val, y_val, batch_size=batch_size):
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# 进行K折交叉验证
for fold, (train_index, val_index) in enumerate(kf.split(features)):
    #print(f'Fold {fold + 1}')
    
    X_train, X_val = features[train_index], features[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    
    train_loader, val_loader = get_data_loader(X_train, y_train, X_val, y_val)
    # model 注册
    model = ResNet1D(num_classes=num_class, dropout=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    model.train()

    if fold == 0:  # 只需在第一个fold上创建背景数据
        background_data = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

    for epoch in range(num_epochs):
        ep_loss = 0.
        wrong_class_nums = 0
        for iteration, (feature, target) in enumerate(train_loader):
            # 计算网络输出
            pred = model(feature)
            
            # 计算损失
            loss = criterion(pred, target)
            ep_loss += loss.item()        
            
            if epoch % test_ep == 0:  
                pred = torch.argmax(pred.detach(), dim=-1)
                target = torch.argmax(target, dim=-1)
                wrong_class_nums += torch.sum(torch.where(pred != target, 1, 0))         

            # 计算梯度和做反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % test_ep == 0: 
            acc = 1 - wrong_class_nums/((iteration+1)*batch_size)  
            print(f'Fold {fold + 1}, Train Acc {acc*100}%')   
        print(f'Fold {fold + 1}, Epoch {epoch+1}, Loss: {ep_loss/(iteration+1)}')

    model.eval()
    wrong_class_nums = 0
    for iteration, (feature, target) in enumerate(val_loader):

        # 计算网络输出
        pred = model(feature)
        pred = torch.argmax(pred, dim=-1)
        target = torch.argmax(target, dim=-1)      


        wrong_class_nums += torch.sum(torch.where(pred != target, 1, 0))
    acc = 1 - wrong_class_nums/((iteration+1)*batch_size)
    acc_list.append(acc)
    acc_sum += acc
    print(f'Fold {fold + 1}, Acc {acc*100}%')   
print (f'Avg acc {acc_sum/n_splits*100}%')   
print (acc_list)   
model.eval()

# 使用SHAP库的DeepExplainer来计算SHAP值
explainer = shap.DeepExplainer(model, torch.tensor(background_data, dtype=torch.float32))
shap_values = explainer.shap_values(torch.tensor(features, dtype=torch.float32),check_additivity=False)

print(shap_values)#/shap_values.mean()


# 打开一个文本文件写入
with open('shap_values_data1.txt', 'w') as f:
    # 遍历每一个切片
    for i in range(shap_values.shape[-1]):
        # 写入切片标题
        f.write(f"Slice {i}:\n")
        # 使用np.savetxt保存每个2D切片
        np.savetxt(f, shap_values[:,:,i], fmt='%.10f', delimiter=',')
        f.write('\n')  # 添加一个空行以分隔切片

# 绘制并保存每个类别的SHAP summary plot
for i in range(num_class):
    #if shap_values[i].shape == features[val_index].shape:
    print(f"Generating and saving SHAP summary plot for class {i}")
    
    # 创建条形图并保存
    plt.figure()
    shap.summary_plot(shap_values[:,:,i], features.numpy(), plot_type="bar", show=False)

    plt.savefig(f"shap_summary_bar_class_{i}.png")
    plt.close()

    # 创建蜜蜂图并保存
    plt.figure()
    shap.summary_plot(shap_values[:,:,i], features.numpy(), show=False)       
    plt.savefig(f"shap_summary_beeswarm_class_{i}.png")
    plt.close()
