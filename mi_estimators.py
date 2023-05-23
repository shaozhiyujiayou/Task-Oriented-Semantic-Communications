import numpy as np
import math

import torch 
import torch.nn as nn


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        # (positive.sum(dim=-1) - negative.sum(dim=-1)) 从正对数似然之和中减去负对数似然之和。
        # negative.sum(dim=-1)，dim=-1 参数指定求和是沿着最后一个维度执行的
        # 这会产生一个标量值，表示总正对数似然和负对数似然之间的差异,导致形状为 [nsample] 的张量
        # 计算正对数似然和负对数似然之间的平均差值，作为 X 和 Y 之间互信息的估计
        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    # loglikeli 方法通过考虑预测均值与真实 y_samples 值之间的平方差除以对数方差的指数，并惩罚大方差来计算非标准化对数似然。 
    # 然后，该方法对每个样本的项求和，并取样本的平均值以获得标量值
    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        # -logvar：此项对应于负对数方差。 它惩罚给定 x_samples 的 y_samples 预测中的大方差
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
    
# CLUBSample 类引入了负对的采样并对估计的互信息应用归一化，以提供对 X 和 Y 之间互信息的更有效和准确的估计    
class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # # x_samples 的形状预计为 [sample_size, x_dim]，其中 sample_size 是样本数，x_dim 是样本的维数
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        
        # 生成从 0 到 sample_size - 1 的随机索引排列
        # torch.randperm(n) 创建从 0 到 n-1 的随机整数排列
        # 调用 .long() 方法将索引的张量转换为 long 数据类型。 这通常是为了确保与代码中使用的其他整数值的一致性
        # 生成random_index张量，为对比学习过程中选择负样本提供随机指标。 这些随机指标确保从可用样本中随机选择负样本
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        # 它不是使用所有可能的对来计算负项，而是通过排列索引 (random_index) 随机采样负样本
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        # # 这个除以2是为了保证估计的互信息落在一个合理的范围内。 它是特定于 CLUB 估计器的采样版本的规范化步骤
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


# MINE 类是实现 MINE（互信息神经估计）算法的 PyTorch 模块
class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        
        # 它采用 x_dim + y_dim 维度的串联输入并应用线性变换以产生 hidden_size 维度的输出。 它执行矩阵乘法，然后添加偏差\
        # 它应用逐元素非线性，保持正值不变并将负值设置为零。 ReLU 将非线性引入网络，使其能够学习输入和输出之间的复杂关系
        # nn.Linear(hidden_size, 1)：这是另一个线性层，它采用前一层的输出（具有 hidden_size 维度）并应用线性变换以产生标量输出。 它执行矩阵乘法，然后添加偏差。
        # 这些线性层和激活函数的目的是学习从 x_dim + y_dim 维度的串联输入到标量输出的映射
        # 通过训练 MINE 模型，神经网络学习捕获输入变量之间的关系并估计它们的互信息
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        
        # torch.randint(sample_size, (sample_size,))：这会生成一个大小为 (sample_size,) 的随机张量，
        # 其中每个元素都是从范围 [0, sample_size) 中随机抽取的整数。 
        # 这意味着张量将包含范围从 0 到 sample_size-1 的唯一随机索引
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]
# 则生成的串联张量的形状为 [sample_size, x_dim + y_dim]。dim=-1 参数指定应该沿着最后一个维度执行串联
        T0 = self.T_func(torch.cat([x_samples,y_samples], dim = -1))
# 将y_samples打乱得到T1的目的是为了在后续计算中与T0进行比较
        T1 = self.T_func(torch.cat([x_samples,y_shuffle], dim = -1))

        
       # 下限计算为 T0 的平均值减去 T1 的指数平均值的对数。 这个下界是 X 和 Y 之间负互信息的近似值，用作训练 MINE 模型的替代损失
       # 它们的均值之间的差异捕获了输入变量之间的关系
        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound
    
    # 返回下界的负值。 这是因为最大化下限等同于最小化负下限
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)

    
class NWJ(nn.Module):   
    def __init__(self, x_dim, y_dim, hidden_size):
        super(NWJ, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
                                    
    def forward(self, x_samples, y_samples): 
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        # [sample_size, sample_size, x_dim] for x_tile 
        #  [sample_size, sample_size, y_dim] for y_tile.
        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        # 通过沿着最后一个维度 (dim=-1) 连接它们，我们创建了一个形状为 [sample_size, sample_size, x_dim + y_dim] 的张量。 
        # 然后这个张量通过 F_func 传递，得到 T1，其形状为 [sample_size, sample_size, 1]
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))-1.  #shape [sample_size, sample_size, 1]

        # 由于 T0 的形状为 [sample_size, 1]，取平均值将得到一个标量值
        # T1.logsumexp(dim = 1)：这将沿张量 T1 的第二维 (dim=1) 应用 logsumexp 函数。 
        # logsumexp 函数用于在数值上稳定指数和的对数计算。 它对指定维度的指数求和，并对结果取对数。 生成的张量的形状为 [sample_size, 1]
        # (T1.logsumexp(dim = 1) - np.log(sample_size))：减去 sample_size 的对数，所得张量的形状为 [sample_size, 1]
        # 由于张量的形状为 [sample_size, 1]，取 平均值将产生标量值
        # 结果是下限估计 这个lower bound 
        lower_bound = T0.mean() - (T1.logsumexp(dim = 1) - np.log(sample_size)).exp().mean() 
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


    
class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(InfoNCE, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1),
                                    nn.Softplus()) # 最后一层Softplus激活的目的是为了保证输出是非负的
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))  #[sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size)) 
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)



# log_sum_exp 函数提供了一种数值稳定的方法来计算指数和的对数，尤其是在处理可能导致数值不稳定的大值或小值时
def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


class L1OutUB(nn.Module):  # naive upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(L1OutUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples): 
        batch_size = y_samples.shape[0]
        mu, logvar = self.get_mu_logvar(x_samples)

        # 从前一项中减去 logvar/2。 该术语解释了对数方差正则化
        # 得到形状为 [nsample] 的张量。 张量中的每个元素代表每个样本的负对数似然项和对数方差正则化项的总和
        positive = (- (mu - y_samples)**2 /2./logvar.exp() - logvar/2.).sum(dim = -1) #[nsample]

        mu_1 = mu.unsqueeze(1)          # [nsample,1,dim]
        logvar_1 = logvar.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)            # [1,nsample,dim]
        all_probs =  (- (y_samples_1 - mu_1)**2/2./logvar_1.exp()- logvar_1/2.).sum(dim = -1)  #[nsample, nsample]

        # 具有负常数值 (-20) 的对角掩码矩阵
        # torch.ones([batch_size]) 创建一个形状为 [batch_size] 的张量。
        # .diag() 通过将所有非对角元素设置为零，将张量转换为对角矩阵。
        # .unsqueeze(-1) 在张量的末尾添加一个额外的维度，导致 [batch_size, 1] 的形状
        # cuda() 将张量移动到 GPU（如果可用）。 这假设模型在 GPU 上运行
        # * (-20.) 将张量按元素乘以 -20，为每个对角线元素分配一个负常量值 (-20)
        # 生成的 diag_mask 张量是形状为 [batch_size, batch_size] 的方阵，其中所有对角线元素都设置为 -20，所有其他元素都为零
        diag_mask =  torch.ones([batch_size]).diag().unsqueeze(-1).cuda() * (-20.)
        
        # all_probs 是形状为 [nsample, nsample] 的张量，包含负样本对的概率之和。
        # diag_mask 是形状为 [batch_size, batch_size] 的张量，所有对角线元素设置为 -20，所有其他元素设置为零。 此掩码用于调整负项的计算。
        # all_probs + diag_mask 将 diag_mask 按元素添加到 all_probs。 这将沿对角线的负样本对的概率增加了 -20。
        # log_sum_exp(all_probs + diag_mask, dim=0) 计算沿第一个维度（维度 0）的每个元素的指数之和的对数。 这通过沿行求和将形状从 [nsample, nsample] 减少到 [nsample]
        # 生成的负张量的形状为 [nsample]
        negative = log_sum_exp(all_probs + diag_mask,dim=0) - np.log(batch_size-1.) #[nsample]
      
        return (positive - negative).mean()
        
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

    
class VarUB(nn.Module):  #    variational upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(VarUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
            
    def forward(self, x_samples, y_samples): #[nsample, 1]
        mu, logvar = self.get_mu_logvar(x_samples)
        return 1./2.*(mu**2 + logvar.exp() - 1. - logvar).mean()
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

    
