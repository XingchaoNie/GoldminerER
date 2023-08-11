import torch
import torch.nn.functional as F 
import numpy as np 
import math 

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ETM(nn.Module):
    def __init__(self, num_topics, vocab_size, t_hidden_size, rho_size, emsize, 
                    theta_act, embeddings=None, train_embeddings=True, enc_drop=0.5):
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emsize = emsize
        self.t_drop = nn.Dropout(enc_drop)

        self.theta_act = self.get_activation(theta_act)
        
        ## define the word embedding matrix \rho
        ## 定义word embedding矩阵
        if train_embeddings:
            #rho_size:300(词向量的维度)
            #vocab_size:3072(输入的文本一共有3072个单词)
            #定义一个linear层。y = wx + b 
            #这个相当于论文中的rho，就是在做word2vec的那个矩阵
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)
            # nn.Linear相当于word2vec的过程。输入维度和输出维度，
        else:
            num_embeddings, emsize = embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            self.rho = embeddings.clone().float().to(device)

        ## define the matrix containing the topic embeddings
        ## 定义包含主题嵌入的矩阵，接受的输入是一个向量，输出的同样是一个向量。垚哥一生吹好吧
        ## self.alphas 和 self.q_theta是论文中要更新的参数，但是现在的问题是。我要怎么更新这两个参数呢？
        ## rho_size: 300 t_hidden_size =800
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)#nn.Parameter(torch.randn(rho_size, num_topics))
    
        ## define variational distribution for \theta_{1:D} via amortizartion
        # 定义variational distribution
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size, t_hidden_size), 
                self.theta_act,
                nn.Linear(t_hidden_size, t_hidden_size),
                self.theta_act,
            ) #这里是一个神经网络。
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)  
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 

    def reparameterize(self, mu, logvar):
        """
        Returns a sample from a Gaussian distribution via reparameterization.
        返回的是采样结果。
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)  # randn_like 生成均值为 0，方差为 1 的正态分布采样的数据，形状和 input 形状相同。
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows):
        """
        Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)  # mu_theta.shape = [1000, 50]  表示每篇文章针对于这50个主题的分布
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def get_beta(self):
        # 函数的作用是得到每个主题对于单词的分布，是一个向量表示呦。
        try:
            logit = self.alphas(self.rho.weight) # torch.mm(self.rho, self.alphas)
            print(self.alphas.weight.shape)  # self.alphas.weight.shape = [50, 300]  每个主题的embedding
            print(self.rho.weight.shape)  # self.rho.weight.shape = [3072, 300]  每个词的embedding
            print(logit.shape)  # logit.shape = [3072, 50]  每个词与每个主题的匹配程度
        except:
            logit = self.alphas(self.rho)
        beta = F.softmax(logit, dim=0).transpose(1, 0) ## softmax over vocab dimension
        return beta

    def get_theta(self, normalized_bows):
        # 函数的作用是得到文档，主题的分布。
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        #mu_theta, logsigma_theta对应着论文里的U_d 和 Sig_d
        z = self.reparameterize(mu_theta, logsigma_theta)
        print('***' * 30)
        print(z)
        print('***' * 30)
        theta = F.softmax(z, dim=-1) 
        return theta, kld_theta

    def decode(self, theta, beta):
        res = torch.mm(theta, beta)   # theta代表了每篇文章和每个主题的匹配程度，beta代表了每个主题和每个词的匹配程度
        #torch.mm() 表示的是矩阵相乘。
        # res.shape = [1000, 3072] 代表了这篇文章生成这个词的概率
        preds = torch.log(res+1e-6)
        return preds 

    def forward(self, bows, normalized_bows, theta=None, aggregate=True):
        ## 肯定是走这个函数，现在的问题是，到底是怎么实现参数的更新的呢？
        ## 这个函数才是核心函数呀，前向传播。
        ## get \theta
        # print('***' * 30)
        # print('\n')
        # print(self.rho.weight)                                                                                                                                                       
        if theta is None:
            theta, kld_theta = self.get_theta(normalized_bows)
        else:
            kld_theta = None
        # print(self.rho.weight)
        
        ## get \beta
        beta = self.get_beta()
        #theta是一个1000行5列的向量。1000行表示的文本的数量，5列表示的是主题的数量。也就是说theta是计算每个文档属于各个主题的概率.
        #beta是一个5行 3072列的向量，5行表示的是主题的数量，3072表示的是文档中单词的数量，beta是各个主题中单词的分布
        ##get prediction loss
        preds = self.decode(theta, beta)
        # print(self.rho.weight)
        #preds 对应着论文中的compute p(w_d| theta_d) = theta_d^T * Beta
        recon_loss = -(preds * bows).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()
        # # print(self.rho.weight)
        # print('\n')
        # print('***' * 30)
        return recon_loss, kld_theta

