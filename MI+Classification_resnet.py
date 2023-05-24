# MI + 基于Resnet的分类
# 使用 CLUB 方法进行互信息 (MI) 估计的 ResNet 模型的训练和评估脚本
import numpy as np
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
import torchvision
import time

from mi_estimators import CLUB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use device :{}".format(device))

#set_seed(1)  # 设置随机种子
label_name = {"airplane": 0, "bird": 1, "car": 2, "cat": 3, "deer": 4, "dog": 5, "horse": 6, "monkey": 7, "ship": 8, "truck": 9}

# 参数设置
MAX_EPOCH = -1
BATCH_SIZE = 64
LR = 0.001
log_interval = 10
val_interval = 1
classes = 10
start_epoch = -1
lr_decay_step = 7
x_dim = 512
y_dim = 512
hidden_size = 15
loop_num = 10
training_steps = 30

Val_Accuracy = []
BEST_VAL_ACC = 0.
# ============================ step 1/5 数据 ============================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(), # 此变换以 0.5 的概率随机水平翻转输入图像。 它有助于引入一些多样性并减少模型对特定方向或方向的依赖
    transforms.ToTensor(),  # 此转换将图像数据从 PIL 图像格式（或其他格式）转换为 PyTorch 张量。 它还将像素值归一化到范围 [0, 1]
#    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize(256), # 此转换将输入图像调整为大小为 256x256 像素的正方形。 保持原始图像的纵横比
    transforms.CenterCrop(224), # 此转换将调整大小后的图像的中心部分裁剪为大小为 224x224 像素的正方形。 图像居中并围绕其中心裁剪
    transforms.ToTensor(), # 此转换将图像数据从 PIL 图像格式（或其他格式）转换为 PyTorch 张量。 它还将像素值归一化到范围 [0, 1]
#    transforms.Normalize(norm_mean, norm_std),
])

train_data = torchvision.datasets.STL10('/home/messor/users/liuchuanhong/liuchuanhong/CNN_classification/source/stl', 'train', transform=train_transform, download=True)
# download=True：该参数表示如果数据集尚未下载，则应从指定路径下载。 如果找不到数据集，会自动下载并保存到指定目录
test_data = torchvision.datasets.STL10('/home/messor/users/liuchuanhong/liuchuanhong/CNN_classification/source/stl', 'test', transform=valid_transform, download=True)
# 构建DataLoder
# batch_size=BATCH_SIZE：此参数确定数据加载器的批量大小。 它指定在每次迭代中加载和处理多少样本
# shuffle=True：该参数表示是否在加载过程中对数据进行shuffle。 将其设置为 True 意味着数据将在每个 epoch 之前随机打乱，这有助于通过在样本顺序中引入随机性来实现更好的训练性能
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
print("length of train_loader: ", len(train_loader))
print("length of valid_loader: ", len(valid_loader))


# ============================ step 2/5 模型 ============================
# 1/3 构建模型
# 创建了 torchvision 库提供的 ResNet-18 模型的实例
# ResNet-18 是一种流行的卷积神经网络 (CNN) 架构，由 18 层组成。 它通常用于图像分类任务
# pretrained=True：此参数指定模型应加载预训练权重。 当 pretrained=True 时，模型使用已在 ImageNet 数据集上训练的权重进行初始化。 
# 这种初始化有助于利用从大型数据集中学习到的特征，这有利于迁移学习
# 通过创建 resnet18_ft 对象，您现在拥有一个在 ImageNet 上预训练的 ResNet-18 模型实例。 
# 您可以针对您的特定任务进一步自定义或微调此模型，例如 STL10 数据集上的图像分类
resnet18_ft = models.resnet18(pretrained=True)

# 3/3 替换fc层
num_ftrs = resnet18_ft.fc.in_features # 此行检索 ResNet-18 模型的输入特征数（最后一个全连接层的输出大小）。 该值将用于确定新的全连接层的输入特征数
resnet18_ft.fc = nn.Linear(num_ftrs, classes) # 此行将 ResNet-18 模型的现有全连接层 (resnet18_ft.fc) 替换为新的全连接层 (nn.Linear(num_ftrs, classes))，该层具有 适当数量的输入特征 (num_ftrs) 和所需数量的输出类 (classes)。
resnet18_ft.to(device) # 将整个ResNet-18模型，包括它的所有参数和缓冲区，移动到指定的设备（device）。 如果 GPU 可用 (cuda)，模型将被移动到 GPU 以加快计算速度。 否则，它将被移动到 CPU

resnet18_ft.to(device)
# chkpt_path = '/home/messor/users/liuchuanhong/liuchuanhong/CLUB-master/resnet_10_classes_15db_mi_10.pth'
# resnet18_ft.load_state_dict(torch.load(chkpt_path))
# ============================ step 3/5 损失函数 ============================
# 定义了分类任务的损失标准
# 模型被训练为将图像分类为十个类别之一，因此 nn.CrossEntropyLoss() 是计算the predicted class probabilities和ground truth labels之间损失的合适选择。 
# 该标准将计算训练期间的损失，可用于通过反向传播更新模型的权重
criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数

# ============================ step 4/5 优化器 ============================
# 法2 : conv 小学习率


# 将 resnet18_ft.parameters() 作为参数传递，您告诉优化器优化模型的可学习参数
# 学习率决定了优化器更新模型参数的步长，它控制模型在训练期间学习的速度
# momentum=0.9：设置优化器的动量值。 动量是一个超参数，可以在相关方向上加速梯度下降算法并抑制振荡。 它有助于优化器更快地收敛并获得稳定的解决方案
optimizer = optim.SGD(resnet18_ft.parameters(), lr=LR, momentum=0.9)               # 选择优化器，用于在训练期间更新 resnet18_ft 模型的参数


# 以下是scheduler程序中使用的参数的细分：
# optimizer：此参数指定scheduler将为其调整学习率的优化器
# step_size：此参数设置学习率降低之前的 epoch 数。 它决定了调整学习率的频率，也就是7个之后降低学习率
# gamma：此参数控制学习率将降低的量。 它在 step_size 个时期后乘以学习率以减少它。 在这种情况下，学习率将降低 0.1 倍
# 这允许更好地控制学习率计划，并有助于提高模型在训练期间的性能
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)     # 设置学习率下降策略，根据指定的时间表在训练期间调整学习率


# 构建MI互信息估计模型
# 在实例化一个名为 mi_model 的 CLUB 模型对象
mi_model = CLUB(x_dim, y_dim, hidden_size).cuda()

# mi_model.parameters()：这指定了将在训练期间更新的 mi_model 的参数。
# LR：这是优化器的学习率，它决定了每次迭代的步长。
# Adam 优化器是一种流行的优化算法，以其在训练神经网络方面的有效性而闻名。 它根据对梯度的一阶和二阶矩的估计来调整每个参数的学习率，从而实现高效稳定的更新
mi_optimizer = torch.optim.Adam(mi_model.parameters(), LR)
mi_est_values = []

# mi_chkpt_path = '/home/messor/users/liuchuanhong/liuchuanhong/CLUB-master/mi_model_resnet_15db.pth'
# mi_model.load_state_dict(torch.load(mi_chkpt_path))
# mi_model.eval()

for loop in range(loop_num):
    resnet18_ft.eval() # 将 ResNet-18 模型 (resnet18_ft) 设置为评估模式。 这样做是为了确保批量归一化和丢失层在评估期间表现正确
    for step in range(training_steps): 
        for i, data in enumerate(train_loader):
            mi_model.train()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
# 输入数据通过 ResNet-18 模型处理直到 avgpool 层，输出存储在 batch_x0 中
            x = inputs
            x = resnet18_ft.conv1(x)# This layer performs convolution on the input tensor to extract features
            x = resnet18_ft.bn1(x)# Batch normalization normalizes the output activations to help with training stability
            x = resnet18_ft.relu(x)#(ReLU) 激活函数按元素应用于批量归一化层的输出。 ReLU 将非线性引入模型
            x = resnet18_ft.maxpool(x)# ReLU激活的输出通过最大池化层（maxpool）。 最大池化减少了特征图的空间维度，同时保留了重要的特征

            x = resnet18_ft.layer1(x)
            x = resnet18_ft.layer2(x)
            x = resnet18_ft.layer3(x)
            x = resnet18_ft.layer4(x)

            # 输出x经过平均池化层（avgpool）。 平均池将特征图的空间维度减少到固定大小。 生成的张量 batch_x0 表示输入特征的summary
            batch_x0 = resnet18_ft.avgpool(x)
            batch_x = batch_x0.view(batch_x0.size(0), -1)
            # channel
            snr = 25     #信噪比
            snr = 10**(snr/10.0)
            xpower = torch.sum(batch_x**2,1)/512.
            npower = xpower/snr
            noise = torch.FloatTensor(512,batch_x.size(0)).to("cuda")
            noise = noise.normal_()*torch.sqrt(npower)
            noise = noise.transpose(1,0)
            batch_y = batch_x + noise
            
            #  计算互信息估计模型的学习损失
            model_loss = mi_model.learning_loss(batch_x, batch_y)
            # print("mi_model_loss: ", model_loss)
            
            mi_optimizer.zero_grad()
            model_loss.backward()
            mi_optimizer.step()
            mi_model.eval()
            mi_value = mi_model(batch_x, batch_y).item()
            print("mi_value: ", mi_value)
            # if i>75:
            #     mi_model.eval()
            #     # mi_est_values.append(mi_model(batch_x, batch_y).item())
            #     print(i,":   mutual information: ", mi_model(batch_x, batch_y).item(), end=" ")
            
            del batch_x, batch_y
            torch.cuda.empty_cache()
        
        if step%10 == 0:
            print("step: ", step)
    print('Find MI Model and Saving it...')
    # torch.save(mi_model.state_dict(),'mi_model_resnet_25db.pth')

    # ============================ step 5/5 训练 ============================

    mi_model.eval()
    print("start training...")
    for epoch in range(start_epoch + 1, MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        resnet18_ft.train()
        for i, data in enumerate(train_loader):

            # forward
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # outputs = resnet18_ft(inputs)

            x = inputs
            x = resnet18_ft.conv1(x)
            x = resnet18_ft.bn1(x)
            x = resnet18_ft.relu(x)
            x = resnet18_ft.maxpool(x)

            x = resnet18_ft.layer1(x)
            x = resnet18_ft.layer2(x)
            x = resnet18_ft.layer3(x)
            x = resnet18_ft.layer4(x)

            batch_x0 = resnet18_ft.avgpool(x)
            batch_x = batch_x0.view(batch_x0.size(0), -1)
            
            # channel
            snr = 25    #信噪比
            snr = 10**(snr/10.0)  
            xpower = torch.sum(batch_x**2,1)/512.
            npower = xpower/snr
            noise = torch.FloatTensor(512,batch_x.size(0)).to("cuda")
            noise = noise.normal_()*torch.sqrt(npower)
            # noise = noise.view(batch_x.size(0), -1)
            noise = noise.transpose(1,0)
            batch_y = batch_x + noise

            outputs = resnet18_ft.fc(batch_y)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels) - 0.0001*mi_model(batch_x, batch_y)
            loss.backward()

            # update weights
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().cpu().sum().numpy()

            # 打印训练信息
            loss_mean += loss.item()
            if (i+1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.


        scheduler.step()  # 更新学习率
        torch.cuda.empty_cache()

        # validate the model
        if (epoch+1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            resnet18_ft.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    x = inputs
                    x = resnet18_ft.conv1(x)
                    x = resnet18_ft.bn1(x)
                    x = resnet18_ft.relu(x)
                    x = resnet18_ft.maxpool(x)

                    x = resnet18_ft.layer1(x)
                    x = resnet18_ft.layer2(x)
                    x = resnet18_ft.layer3(x)
                    x = resnet18_ft.layer4(x)

                    batch_x0 = resnet18_ft.avgpool(x)
                    batch_x = batch_x0.view(batch_x0.size(0), -1)
                    
                    # channel
                    snr = 25     #信噪比
                    snr = 10**(snr/10.0)  
                    xpower = torch.sum(batch_x**2,1)/512.
                    npower = xpower/snr
                    noise = torch.FloatTensor(512,batch_x.size(0)).to("cuda")
                    noise = noise.normal_()*torch.sqrt(npower)
                    # noise = noise.view(batch_x.size(0), -1)
                    noise = noise.transpose(1,0)
                    batch_y = batch_x + noise

                    outputs = resnet18_ft.fc(batch_y)

                    # outputs = resnet18_ft(inputs)
                    loss = criterion(outputs, labels) - 0.0001*mi_model(batch_x, batch_y)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

                    loss_val += loss.item()

                loss_val_mean = loss_val/len(valid_loader)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val_mean, correct_val / total_val))
            resnet18_ft.train()

            acc = correct_val / total_val
            Val_Accuracy.append(acc)
            # if acc > BEST_VAL_ACC:
            #     print('Find Better Model and Saving it...')
            #     torch.save(resnet18_ft.state_dict(),
            #             'resnet_10_classes_25db_mi_12.pth')
            #     BEST_VAL_ACC = acc
            #     print('Saved!')
    resnet18_ft.eval()
    if loop%5==0:
        print("loop: ", loop)


