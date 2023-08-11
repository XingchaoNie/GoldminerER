#/usr/bin/python

from __future__ import print_function

import argparse
import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
import sys
import matplotlib.pyplot as plt 
import data
import utils
import scipy.io
from torch import nn, optim
from torch.nn import functional as F

from etm import ETM
# from utils import nearest_neighbors, get_topic_coherence, get_topic_diversity

parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
parser.add_argument('--dataset', type=str, default='20ng', help='name of corpus')
parser.add_argument('--data_path', type=str, default='data/20ng/', help='directory containing data')
parser.add_argument('--emb_path', type=str, default='data/20ng_embeddings.txt', help='directory containing word embeddings')
parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for training')

### model-related arguments
parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--train_embeddings', type=int, default=1, help='whether to fix rho or train it')

### optimization-related arguments
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train...150 for 20ng 100 for others')
parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 1)')
parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')
parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=10, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=2, help='when to log training')
parser.add_argument('--visualize_every', type=int, default=10, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
parser.add_argument('--tc', type=int, default=0, help='whether to compute topic coherence or not')
parser.add_argument('--td', type=int, default=0, help='whether to compute topic diversity or not')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(args) #输出所有的参数
# print(device) #输出使用的设备名，如果有显卡，那么使用GPU，否则使用cpu
print('\n')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

## get data
# 1. vocabulary
vocab, train, valid, test = data.get_data(os.path.join(args.data_path)) #读入数据，vocab词典的长度，trian,test分别是训练集和测试集,valid应该是验证集


# print(vocab) #所有的单词组成的数组
#print(len(vocab)) #一共有3072个
# print(train)
# train 是一个字典。有两个元素，一个是tokens 一个是counts。 counts表示的是一共有多少个tokens,然而 tokens表示的是啥，目前还不知道。
# 哦哦，知道了，tokens表示的是文档的内容，只不过是用数字表示出来的。即每一篇文档都是一个二维数组，数组的内容就是单词对应的序号。
# 下面的test和valid也和这个train类似。一个的是测试集，一个是验证集。
#print(valid)
# print(test)
# print(train['tokens'])
vocab_size = len(vocab) #计算字典的长度。
args.vocab_size = vocab_size

# 1. training data
train_tokens = train['tokens']
train_counts = train['counts']
args.num_docs_train = len(train_tokens)

# 2. dev set
valid_tokens = valid['tokens']
valid_counts = valid['counts']
args.num_docs_valid = len(valid_tokens)

# 3. test data
test_tokens = test['tokens']
test_counts = test['counts']
args.num_docs_test = len(test_tokens)
test_1_tokens = test['tokens_1']
test_1_counts = test['counts_1']
args.num_docs_test_1 = len(test_1_tokens)
test_2_tokens = test['tokens_2']
test_2_counts = test['counts_2']
args.num_docs_test_2 = len(test_2_tokens)

embeddings = None
if not args.train_embeddings:
    # 我们给的输入是包含train_embeddings的，所以这个if语句是不会走进来的。
    # 问题是这个train_embeddings的含义是啥呀??? 2020-3-13 18:56
    emb_path = args.emb_path
    vect_path = os.path.join(args.data_path.split('/')[0], 'embeddings.pkl')   
    vectors = {}
    with open(emb_path, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            if word in vocab:
                vect = np.array(line[1:]).astype(np.float)
                vectors[word] = vect
    embeddings = np.zeros((vocab_size, args.emb_size))
    words_found = 0
    for i, word in enumerate(vocab):
        try: 
            embeddings[i] = vectors[word]
            words_found += 1
        except KeyError:
            embeddings[i] = np.random.normal(scale=0.6, size=(args.emb_size, ))
    embeddings = torch.from_numpy(embeddings).to(device)
    args.embeddings_dim = embeddings.size()

# print('=*'*100)
# print('Training an Embedded Topic Model on {} with the following settings: {}'.format(args.dataset.upper(), args))
# print('=*'*100)

## define checkpoint
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.mode == 'eval':
     ckpt = args.load_from
else:
    #ckpt 是保存模型的位置。
    ckpt = os.path.join(args.save_path, 
        'etm_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}'.format(
        args.dataset, args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act, 
            args.lr, args.batch_size, args.rho_size, args.train_embeddings))

## 初始化模型，定义优化器（优化器使用一般是adam,这个我熟，我还手写过logisticRegression的adam优化算法呢。这算个小彩蛋吧。你要是看到了。可以私信我要adam版的LR代码呦。）
model = ETM(args.num_topics, vocab_size, args.t_hidden_size, args.rho_size, args.emb_size, 
                args.theta_act, embeddings, args.train_embeddings, args.enc_drop).to(device)

# print('model: {}'.format(model)) #观察模型中的参数。
print(args)

#选择优化器。
if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'asgd':
    optimizer = optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
else:
    print('Defaulting to vanilla SGD')
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

def train(epoch):
    #训练模型代码
    #哎，是逃不过的。整个论文里最核心的代码来了。。。
    #再读一遍。这已经是第7遍了。还是没看懂。真是让人头大。
    model.train() #这是pytorch自带的函数，model.train() 和 model.eval() 一般在模型训练和评价的时候会加上这两句，主要是针对由于model 在训练时和评价时 Batch Normalization 和 Dropout 方法模式不同；因此，在使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval；
    acc_loss = 0
    acc_kl_theta_loss = 0
    cnt = 0
    indices = torch.randperm(args.num_docs_train) # indices : args.num_docs_train个数打乱成为一个序列
    # args.num_docs_train: 训练文本的个数，一共有11214个文本.
    indices = torch.split(indices, args.batch_size) #将训练文本切割，每一块都有args.batch_size个文本,如果不能整除，那么最后一块的文本数量会小一些,是一个数组
    # print(indices)
    for idx, ind in enumerate(indices):
        # Ques：每次循环，都更新了什么呀？我怎么感觉每次循环都什么都没变呀。。
        # 对文档中的内容遍历
        # 对应论文中的choose a minibatch B of documents
        # idx：第几块文本 ind:每一块文本的内容。
        optimizer.zero_grad()
        model.zero_grad()
        #两个函数的作用都是模型中参数的梯度设置成0
        data_batch = data.get_batch(train_tokens, train_counts, ind, args.vocab_size, device)
        # data_batch 是1000 * 3072的向量，1000是文本的个数。3072是一个单词的个数
        # 每个文本是用3072个单词表示的。
        # 一次训练1000个文本。
        sums = data_batch.sum(1).unsqueeze(1)
        # tensor.sum(1) : 按行求和
        # tensor.sum(0) : 按列求和
        # tensor.unsqueeze : 对data_batch进行扩充，在这段代码里的作用就是将原本的一维数组，转成2维数组。
        if args.bow_norm:
            normalized_data_batch = data_batch / sums
            # 做归一化处理, 使得data_batch每一行的和都是1
            # Get normalized bag-of-word representat x_d
        else:
            normalized_data_batch = data_batch
        # print('***' * 20)
        # print('\n')
        # print(data_batch.shape)
        # print(normalized_data_batch.shape)
        # data_batch原始向量，normalized_data_batch正则化后的向量。
        recon_loss, kld_theta = model(data_batch, normalized_data_batch) # 也不是一点收获没有。最起码知道，这两个东西是两个tensor，好,下面的问题是。这两个东西是怎么算的。2020-3-14 22:30
        #是在这步更新的参数
        #这里调用的是model里forward函数。
        #tensor(612.6284, device='cuda:0', grad_fn=<MeanBackward0>) tensor(0.1139, device='cuda:0', grad_fn=<MulBackward0>)
        # print(recon_loss, kld_theta)
        # print('\n')
        # print('***' * 20)
        
        total_loss = recon_loss + kld_theta
        total_loss.backward()
        # print('Q.Q' * 20) #这代码看的这绝望呀。。  Q ^ Q 👈这个表情就是我现在的样子
        #                   #👆这才哪到哪呀。论文更绝望。2020-3-13 17：39
        #                   # 2020-3-13 20：02 我吃过晚饭，去外面溜达一圈，又回来了，北京的晚上有点冷。楼下好多人都带个口罩在遛弯。估计都被憋坏了。
        # print('\n')
        # print(recon_loss, kld_theta)
        # print(total_loss.backward())
        # 2020-3-15 13：32 现在就剩下一个问题了。在代码中，哪里体现了更新模型参数和variational paramenters。这两个参数。
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            #梯度剪切，规定了最大不能超过的args.clip
        optimizer.step() #这行代码实现了对参数的更新。
        acc_loss += torch.sum(recon_loss).item()  
        acc_kl_theta_loss += torch.sum(kld_theta).item()
        cnt += 1

        if idx % args.log_interval == 0 and idx > 0:
            #round是实现四舍五入的。
            cur_loss = round(acc_loss / cnt, 2) # 
            cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
            cur_real_loss = round(cur_loss + cur_kl_theta, 2) # Estimate the ELBO and its gradient(backporp)
            #这三个参数应该是计算目前我看不懂那个 Variational inference.

            print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, idx, len(indices), optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
    
    cur_loss = round(acc_loss / cnt, 2) 
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
    cur_real_loss = round(cur_loss + cur_kl_theta, 2)
    print('*'*100)
    print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
            epoch, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
    print('*'*100)

def visualize(m, show_emb=True):
    #可视化模型
    #如何根据代码中的变量来对文章提取主题。
    #整篇论文的目的，就是求得self.rho和beta。再对这论文看一次代码。
    if not os.path.exists('./results'):
        os.makedirs('./results')

    m.eval()

    queries = ['andrew', 'computer', 'sports', 'religion', 'man', 'love', 
                'intelligence', 'money', 'politics', 'health', 'people', 'family']

    ## 可视化主题
    with torch.no_grad():
        print('#'*100)
        print('Visualize topics...')
        topics_words = []
        gammas = m.get_beta() # 也就是m.get_beta()的作用是得到主题关于此的分布。这也是个向量。
        #现在的理解是gammas每个主题的向量。5 * 3072维的。5行是有五个主题，3072是单词的个数。
        for k in range(args.num_topics):
            gamma = gammas[k]
            top_words = list(gamma.cpu().numpy().argsort()[-args.num_words+1:][::-1]) #对所得到的词排序。
            topic_words = [vocab[a] for a in top_words]
            topics_words.append(' '.join(topic_words))
            print('Topic {}: {}'.format(k, topic_words))

        if show_emb:
            ## visualize word embeddings by using V to get nearest neighbors
            ## 展示每个单词的上下文。使用上下文来做CBOW来实现对word 2 vec
            print('#'*100)
            print('Visualize word embeddings by using output embedding matrix')
            # 输出中有一个vectors : (3072, 300) 和 query:(300,)这两个东西，是从哪来的。我知道这个元组的意义是3072的输入向量转成300维的词向量。
            try:
                embeddings = m.rho.weight  # Vocab_size x E
            except:
                embeddings = m.rho         # Vocab_size x E
            # embeddings表示的是词向量的维度，3072 * 300
            neighbors = []
            for word in queries:
                print('word: {}    neighbors: {}'.format(
                    word, utils.nearest_neighbors(word, embeddings, vocab)))
            print('#'*100)

def evaluate(m, source, tc=False, td=False):
    """
    评估模型的好坏
    Compute perplexity on document completion.
    """
    m.eval()
    with torch.no_grad():
        if source == 'val':
            indices = torch.split(torch.tensor(range(args.num_docs_valid)), args.eval_batch_size)
            tokens = valid_tokens
            counts = valid_counts
        else: 
            indices = torch.split(torch.tensor(range(args.num_docs_test)), args.eval_batch_size)
            tokens = test_tokens
            counts = test_counts

        ## get \beta here
        beta = m.get_beta()

        ### do dc and tc here
        acc_loss = 0
        cnt = 0
        indices_1 = torch.split(torch.tensor(range(args.num_docs_test_1)), args.eval_batch_size)
        for idx, ind in enumerate(indices_1):
            ## get theta from first half of docs
            data_batch_1 = data.get_batch(test_1_tokens, test_1_counts, ind, args.vocab_size, device)
            sums_1 = data_batch_1.sum(1).unsqueeze(1)
            if args.bow_norm:
                normalized_data_batch_1 = data_batch_1 / sums_1
            else:
                normalized_data_batch_1 = data_batch_1
            theta, _ = m.get_theta(normalized_data_batch_1)

            ## get prediction loss using second half
            data_batch_2 = data.get_batch(test_2_tokens, test_2_counts, ind, args.vocab_size, device)
            sums_2 = data_batch_2.sum(1).unsqueeze(1)
            res = torch.mm(theta, beta)
            preds = torch.log(res)
            recon_loss = -(preds * data_batch_2).sum(1)
            
            loss = recon_loss / sums_2.squeeze()
            loss = loss.mean().item()
            acc_loss += loss
            cnt += 1
        cur_loss = acc_loss / cnt
        ppl_dc = round(math.exp(cur_loss), 1)
        print('*'*100)
        print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
        print('*'*100)
        if tc or td:
            beta = beta.data.cpu().numpy()
            if tc:
                print('Computing topic coherence...')
                utils.get_topic_coherence(beta, train_tokens, vocab)
            if td:
                print('Computing topic diversity...')
                utils.get_topic_diversity(beta, 25)
        return ppl_dc
        

if args.mode == 'train':
    ## 如果模式是训练的话，走下面的逻辑。
    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []
    # print('\n')
    print('Visualizing model quality before training...')
    visualize(model) # 对模型进行可视化处理
    print('\n')
    for epoch in range(1, args.epochs):
        #从这里开始进行epochs，会进行epochs-1次迭代。
        train(epoch)
        val_ppl = evaluate(model, 'val')
        if val_ppl < best_val_ppl:
            with open(ckpt, 'wb') as f:
                torch.save(model, f)
            best_epoch = epoch
            best_val_ppl = val_ppl
        else:
            ## check whether to anneal lr
            lr = optimizer.param_groups[0]['lr']
            if args.anneal_lr and (len(all_val_ppls) > args.nonmono and val_ppl > min(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
                optimizer.param_groups[0]['lr'] /= args.lr_factor
        if epoch % args.visualize_every == 0:
            #如果模型是可以整除这个数的。那么就显示模型。
            # print('xxxxxx')
            visualize(model)
        all_val_ppls.append(val_ppl)
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    val_ppl = evaluate(model, 'val')
    utils.get_topic_findding()
else:   
    #如果模式是其他的话。走下面的逻辑。哈哈哈哈，可真是个好消息，如果只是想入门ETM的话，剩下的60+行代码都不用看了。开心到飞起(*^▽^*)
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        ## get document completion perplexities
        test_ppl = evaluate(model, 'test', tc=args.tc, td=args.td)
        
        ## get most used topics
        indices = torch.tensor(range(args.num_docs_train))
        indices = torch.split(indices, args.batch_size)
        thetaAvg = torch.zeros(1, args.num_topics).to(device)
        thetaWeightedAvg = torch.zeros(1, args.num_topics).to(device)
        cnt = 0
        for idx, ind in enumerate(indices):
            data_batch = data.get_batch(train_tokens, train_counts, ind, args.vocab_size, device)
            sums = data_batch.sum(1).unsqueeze(1)
            cnt += sums.sum(0).squeeze().cpu().numpy()
            if args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            theta, _ = model.get_theta(normalized_data_batch)
            thetaAvg += theta.sum(0).unsqueeze(0) / args.num_docs_train
            weighed_theta = sums * theta
            thetaWeightedAvg += weighed_theta.sum(0).unsqueeze(0)
            if idx % 100 == 0 and idx > 0:
                print('batch: {}/{}'.format(idx, len(indices)))
        thetaWeightedAvg = thetaWeightedAvg.squeeze().cpu().numpy() / cnt
        print('\nThe 10 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:10]))

        ## show topics
        beta = model.get_beta()
        topic_indices = list(np.random.choice(args.num_topics, 10)) # 10 random topics
        print('\n')
        for k in range(args.num_topics):#topic_indices:
            gamma = beta[k]
            top_words = list(gamma.cpu().numpy().argsort()[-args.num_words+1:][::-1])
            topic_words = [vocab[a] for a in top_words]
            print('Topic {}: {}'.format(k, topic_words))

        if args.train_embeddings:
            ## show etm embeddings 
            try:
                rho_etm = model.rho.weight.cpu()
            except:
                rho_etm = model.rho.cpu()
            queries = ['andrew', 'woman', 'computer', 'sports', 'religion', 'man', 'love', 
                            'intelligence', 'money', 'politics', 'health', 'people', 'family']
            print('\n')
            print('ETM embeddings...')
            for word in queries:
                print('word: {} .. etm neighbors: {}'.format(word, utils.nearest_neighbors(word, rho_etm, vocab)))
            print('\n')
