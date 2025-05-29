# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import pathlib
from collections import OrderedDict, abc as container_abcs
from sklearn.model_selection import StratifiedKFold  # 使用 StratifiedKFold
from sklearn.metrics import roc_curve, auc  # 导入ROC相关库
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from models import ResNet1D
from tqdm import tqdm
import shap
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# 设置画布大小
plt.figure(figsize=(14, 6))  # 10 英寸宽，6 英寸高
# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'

# 读取CSV文件

num_feats = 15
data_path = '202503ResNet.csv'  # 'data.csv'
f = open(data_path, "rb")  # 二进制格式读文件
i = 0
while True:
    i += 1
    line = f.readline()  # 按行读取

    if not line:
        break
    else:
        try:
            line.decode('ANSI')
        except:  # 打印出不能通过'ANSI'方式解码的数据行
            print(i)
            print(str(line))

df = pd.read_csv(data_path, encoding='ANSI')
# 提取样本ID、标签和特征
sample_ids = df.iloc[:, 0].values
print(sample_ids)
labels_input = df.iloc[:, 2].values
cohort = df.iloc[:, 1].values
features = df.iloc[:, 4:].values
cohort = cohort.astype(float)
features = features.astype(float)  # 转换为浮点数
N_samples = len(sample_ids)
label_set = set(labels_input)
#print(label_set)
num_class = len(label_set)
str_to_num = dict((c, i) for i, c in enumerate(label_set))
print(str_to_num)
label_nums = torch.tensor([str_to_num[str_f] for str_f in labels_input])
labels = F.one_hot(label_nums)

# 处理特征数据
print(features.shape)
features = torch.tensor(features)

# 超参数设置
batch_size = 32
num_epochs = 40 
test_ep = 10
lr_init = 0.0001 ########
criterion = torch.nn.CrossEntropyLoss()

# 使用 StratifiedKFold
n_splits = 9  # 折叠数
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 数据集分割函数
def get_data_loader(X_train, y_train, X_val, y_val, batch_size=batch_size):
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))  # 使用 torch.long 类型
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader



# 准备数据进行交叉验证
acc_list = []  # 用于存储每个折叠的准确度
acc_sum = 0  # 累加准确度

# 将 one-hot 编码的标签转回整数标签
labels_int = torch.argmax(labels, dim=1)  # 每个样本的标签

# 用于绘制ROC曲线
all_fpr = np.linspace(0, 1, 100)  # 用于插值
mean_tpr = np.zeros((num_class, len(all_fpr)))  # 存储每个类别的平均TPR
mean_fpr = np.zeros_like(all_fpr)  # 存储平均的FPR

# 记录每个类别的FPR和TPR
tpr_list = [[] for _ in range(num_class)]
fpr_list = [[] for _ in range(num_class)]

model = ResNet1D(num_classes=num_class, dropout=True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
for fold, (train_index, val_index) in enumerate(kf.split(features[cohort==1], labels_int[cohort==1])):  # StratifiedKFold 会自动根据 labels 划分
    # 划分训练集和验证集
    X_train, X_val = features[train_index], features[val_index]
    y_train, y_val = labels_int[train_index], labels_int[val_index]

    # 创建数据加载器
    train_loader, val_loader = get_data_loader(X_train, y_train, X_val, y_val)

    # 初始化模型
    
    model.train()

    if fold == 0:  # 只需在第一个fold上创建背景数据
        background_data = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

    # 训练模型
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
                wrong_class_nums += torch.sum(pred != target)         

            # 计算梯度和做反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % test_ep == 0: 
            acc = 1 - wrong_class_nums / ((iteration + 1) * batch_size)  
            print(f'Fold {fold + 1}, Train Acc {acc * 100}%')               
        print(f'Fold {fold + 1}, Epoch {epoch + 1}, Loss: {ep_loss / (iteration + 1)}')
    acc_sum += acc
    acc_list.append(acc.item())
# 输出平均准确度
print(f'Avg acc {acc_sum / n_splits * 100}%')   
print(acc_list)
model.eval()



# 在验证集上评估模型
y_true = []
y_pred_prob = []
test_dataset = TensorDataset(torch.tensor(features[cohort==2], dtype=torch.float32), torch.tensor(labels_int[cohort==2], dtype=torch.long))  
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for iteration, (feature, target) in enumerate(test_loader):
    # 获取模型的概率输出（softmax）
    pred = model(feature)
    prob = F.softmax(pred, dim=-1).detach().numpy()  # softmax转换为概率
    y_true.append(target.numpy())
    y_pred_prob.append(prob)

y_true = np.concatenate(y_true)
y_pred_prob = np.concatenate(y_pred_prob)
y_pred = np.argmax(y_pred_prob, axis=-1)
acc = (y_pred == y_true).sum() / y_true.shape[0]
print("Full acc = ", acc, y_pred_prob.shape, sample_ids.shape)
combined = np.concatenate((sample_ids.reshape(-1,1).astype(str), y_pred_prob.astype(str)), -1)
np.savetxt('out.txt',combined,fmt='%s')
print(str_to_num)
#top20 retrain数据集

# 使用SHAP库的DeepExplainer来计算SHAP值
explainer = shap.DeepExplainer(model, torch.tensor(background_data, dtype=torch.float32))
shap_values = explainer.shap_values(torch.tensor(features, dtype=torch.float32),check_additivity=False)

#print(shap_values,shap_values.shape)#/shap_values.mean()

global_importance = np.abs(shap_values).mean(0).sum(-1)
#print(global_importance,global_importance.shape)#/shap_values.mean()
inds = np.argsort(-global_importance)
f = plt.figure()
y_pos = np.arange(num_feats) #######
inds2 = np.flip(inds[:num_feats],0) ##### 
top_index = inds2[0]

# 绘制条形图
plt.barh(y_pos, global_importance[inds2], align='center', color='#1ecbe5')  # 修改颜色为#1ecbe5
plt.yticks(y_pos, fontsize=13)
plt.gca().set_yticklabels(df.columns[inds2 + 4])
plt.xlabel('Mean abs. SHAP value', fontsize=13)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('none')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# 保存图表为PNG文件
plt.savefig(f"shap_global_importance.png")

# 保存图表为PDF文件
plt.savefig(f"shap_global_importance.pdf")

plt.close()

print(shap_values.shape, df.columns.shape, features.shape)
# 绘制并保存每个类别的SHAP summary plot
#for i in range(num_class):
for name, i in str_to_num.items():
    #if shap_values[i].shape == features[val_index].shape:
    print(f"Generating and saving SHAP summary plot for class {i}")
    
    # 创建条形图并保存
    plt.figure()
    shap.summary_plot(shap_values[:,:15,i], features[:,:15].numpy(), feature_names=df.columns[4:19], plot_type="bar", show=False)
    plt.savefig(f"shap_summary_bar_class_{name}.pdf")
    plt.close()

    # 创建蜜蜂图并保存
    plt.figure()
    shap.summary_plot(shap_values[:,:15,i], features[:,:15].numpy(), feature_names=df.columns[4:19], show=False)       
    plt.savefig(f"shap_summary_beeswarm_class_{name}.pdf")
    plt.close()


    plt.figure()
    shap.dependence_plot(top_index,shap_values[:,:,i], features.numpy())
    plt.savefig(f"shap_summary_top_1_feat_{name}.pdf")
    plt.close()


acc_list = []  # 用于存储每个折叠的准确度
acc_sum = 0  # 累加准确度
top20_feat = features[...,inds2.copy()]
print(features.shape, top20_feat.shape)
for fold, (train_index, val_index) in enumerate(kf.split(top20_feat[cohort==1], labels_int[cohort==1])):  # StratifiedKFold 会自动根据 labels 划分
    # 划分训练集和验证集
    X_train, X_val = top20_feat[train_index], top20_feat[val_index]
    y_train, y_val = labels_int[train_index], labels_int[val_index]

    # 创建数据加载器
    train_loader, val_loader = get_data_loader(X_train, y_train, X_val, y_val)

    # 初始化模型
    model = ResNet1D(num_classes=num_class, dropout=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    model.train()

    # 训练模型
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
                wrong_class_nums += torch.sum(pred != target)         

            # 计算梯度和做反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % test_ep == 0: 
            acc = 1 - wrong_class_nums / ((iteration + 1) * batch_size)  
            #print(f'Top20 feature Fold {fold + 1}, Train Acc {acc * 100}%')   
            
        print(f'Top20 feature Fold {fold + 1}, Epoch {epoch + 1}, Loss: {ep_loss / (iteration + 1)}')
    acc_sum += acc
    acc_list.append(acc)
# 输出平均准确度
print(f'Avg acc {acc_sum / n_splits * 100}% of top20 feature')   
print(acc_list)
model.eval()

# 在验证集上评估模型top20
y_true = []
y_pred_prob = []
test_dataset = TensorDataset(torch.tensor(top20_feat[cohort==2], dtype=torch.float32), torch.tensor(labels_int[cohort==2], dtype=torch.long))  
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for iteration, (feature, target) in enumerate(test_loader):
    # 获取模型的概率输出（softmax）
    pred = model(feature)
    prob = F.softmax(pred, dim=-1).detach().numpy()  # softmax转换为概率
    y_true.append(target.numpy())
    y_pred_prob.append(prob)

y_true = np.concatenate(y_true)
y_pred_prob = np.concatenate(y_pred_prob)
y_pred = np.argmax(y_pred_prob, axis=-1)
acc = (y_pred == y_true).sum() / y_true.shape[0]
print("Top20 acc = ", acc)
print(y_pred_prob)


# 对每个类别计算ROC曲线并保存为PDF
for i in range(num_class):
    prob = y_pred_prob[:, i]
    fpr, tpr, _ = roc_curve(y_true == i, prob)  # 对于每个类别，视作二分类计算ROC
    roc_auc = auc(fpr, tpr)  # 计算AUC

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC Class {i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Top20 ROC Curve for Class {i}')
    plt.legend(loc='lower right')
    plt.savefig(f"Top20 ROC_curve_class_{i}.pdf")  # 保存为PDF
    plt.close()

# 计算宏平均ROC曲线并保存为PDF
all_fpr = np.linspace(0, 1, 100)  # 用于插值的FPR网格
mean_tpr = np.zeros_like(all_fpr)  # 初始化mean_tpr数组

for i in range(num_class):
    fpr, tpr, _ = roc_curve(y_true == i, y_pred_prob[:, i])  # 每个类别的ROC曲线
    tpr_interpolated = np.interp(all_fpr, fpr, tpr)  # 插值
    mean_tpr += tpr_interpolated

mean_tpr /= num_class  # 计算平均TPR

# 绘制宏平均ROC曲线
plt.figure()
plt.plot(all_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {auc(all_fpr, mean_tpr):.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-Average Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig("Top20 Validation_Average_ROC_curve.pdf")  # 保存为PDF
plt.close()

# 输出完成信息
print("验证集的ROC曲线已生成并保存为PDF文件！")
