# coding:utf-8
import torch
import torch.nn.functional as F
import re
from torch.utils.data import Dataset, DataLoader
import time
import torch.optim as optim
import time
import random
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#开始标志
SOS_token = 0
#结束标志
EOS_token = 1
#最大句子长度
MAX_LENGTH = 11

#文件路径
eng2fre = './eng-fra-v2.txt'

#定义清洗数据工具函数
def normalizerString(s):
    #字符串规范化
    s = s.lower().strip()
    #.!?的前面加上空格
    s = re.sub(r'([.!?])', r' \1 ', s)
    # print(s)
    s = re.sub(r'[^a-zA-Z.!?]+',r' ',s)
    # print(s)
    return s

#读取数据
def my_getdata():
    '''
    从文件读取数据
    :return:
    '''
    with open(eng2fre, encoding='utf-8') as f:
        all_data = f.read()
    my_lines = all_data.strip().split('\n')
    # print(len(my_lines))
    my_pairs = [[normalizerString(s) for s in l.split('\t')] for l in my_lines]
    # print(my_pairs)
    # print(len(my_pairs))
    #遍历语言对，构建英法单词词典
    english_word2index = {'SOS':0,'EOS':1}
    english_word_n = 2

    french_word2index = {'SOS':0,'EOS':1}
    french_word_n = 2

    #
    for pair in my_pairs:
        # print(pair)
        for word in pair[0].split(' '):
            if word not in english_word2index:
                # english_word2index[word] = len(english_word2index)
                english_word2index[word] = english_word_n
                english_word_n += 1
        for word in pair[1].split(' '):
            if word not in french_word2index:
                # french_word2index[word] = len(french_word2index)
                french_word2index[word] = french_word_n
                french_word_n += 1

    # print(len(english_word2index))
    # print(len(french_word2index))
    # print(english_word2index)

    #构建english_index2word french_index2word
    english_index2word = {v:k for k,v in english_word2index.items()}
    french_index2word = {v:k for k,v in french_word2index.items()}
    return english_word2index,english_index2word,english_word_n,french_word2index,french_index2word,french_word_n,my_pairs

#第二步：构建数据源
class MyPairsDataset(Dataset):
    def __init__(self,my_pairs):
        super(MyPairsDataset, self).__init__()
        self.my_pairs = my_pairs
        self.sample_len = len(my_pairs)

    def __len__(self):
        return self.sample_len
    def __getitem__(self,index):
        #异常值修正
        index = min(max(index,0),self.sample_len-1)
        #根据索引取数据
        x = self.my_pairs[index][0]
        y = self.my_pairs[index][1]
        #x y 数值化
        x = [english_word2index[word] for word in x.split(' ')]
        x.append(EOS_token)
        tensor_x = torch.tensor(x,dtype=torch.long,device=device)

        y = [french_word2index[word] for word in y.split(' ')]
        y.append(EOS_token)
        tensor_y = torch.tensor(y, dtype=torch.long, device=device)

        return tensor_x ,tensor_y

#侧式数据源
def dm_test_MyPairsFataset():
    mydataset = MyPairsDataset(my_pairs)
    mydataloader = DataLoader(dataset=mydataset,batch_size=1,shuffle=True)
    for i , (x,y) in enumerate(mydataloader):
        print('x.shape',x.shape,x)
        print('y.shape',y.shape,y)
        break



#定义编码器
class EncoderGRU(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        #实例化nn.Embedding层
        self.embedding = nn.Embedding(vocab_size,embed_size)
        #实例化nn.GRU层
        self.gru = nn.GRU(embed_size,hidden_size,batch_first=True)

    def forward(self,input,hidden):
        #数据经过nn.Embedding层
        output = self.embedding(input)
        #数据经过gru层
        output,hidden = self.gru(output,hidden)
        return output,hidden
    def inithidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)

def dm_test_EncoderGRU():
    mydataset = MyPairsDataset(my_pairs)
    mydataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)
    vocab_size = english_word_n
    embed_size = 256
    hidden_size = 512
    my_encodergru = EncoderGRU(vocab_size,embed_size,hidden_size).to(device)

    for i ,(x,y) in enumerate(mydataloader):
        hidden = my_encodergru.inithidden()
        #一次性
        output,hidden = my_encodergru(x,hidden)
        # print(output.shape)
        # print(hidden.shape)


        #一个字符一个字符
        for i in range(x.shape[1]):
            tmp = x[0][i].view(1,-1)
            output,hidden = my_encodergru(tmp,hidden)
            break
        # print('output',output)
        break

#定义无attention的解码器
class DecoderGRU(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size):
        super().__init__()
        #vocab:法语单词数 #embed词嵌入维度 hidden输出维度
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        #定义embedding
        self.embed = nn.Embedding(self.vocab_size,self.embed_size)
        #GRU
        self.gru = nn.GRU(self.embed_size,hidden_size,batch_first=True)
        #全连接层
        self.linear = nn.Linear(self.hidden_size,self.vocab_size)
        #softmax
        self.softmax = nn.LogSoftmax(dim=-1)
    def forward(self,input,hidden):
        output = self.embed(input)
        output = F.relu(output)#防止过拟合
        output,hidden = self.gru(output,hidden)
        output = self.softmax(self.linear(output[0]))
        return output,hidden

    def inithidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)

#调用
def dm03_test_DecoderGRU():
    mydataset = MyPairsDataset(my_pairs)
    mydataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)
    #实例化encoder模型
    vocab_size = english_word_n
    embed_size = 256
    hidden_size = 512
    my_encodergru = EncoderGRU(vocab_size,embed_size,hidden_size).to(device)
    #实例化decoder模型
    vocab_size = french_word_n
    embed_size = 256
    hidden_size = 512
    my_decodergru = DecoderGRU(vocab_size,embed_size,hidden_size).to(device)

    for i,(x,y) in enumerate(mydataloader):#给模型喂数据
        #编码
        hidden = my_encodergru.inithidden()
        encode_output_c,hidden = my_encodergru(x,hidden)
        #解码
        for i in range(y.shape[1]):
            tmp = y[0][i].view(1,-1)
            output,hidden = my_decodergru(tmp,hidden)
            print(output.shape)
            break
        break

class AttenDecoderGRU(nn.Module):
    def __init__(self,vocab_size,embed_size,encoder_hidden_size,gru_hidden_size,
                 p_dropout=0.1):
        super().__init__()
        #v:法语词汇 em 此维度 encoder_hidden 隐藏层输出维度
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.encoder_hidden_size = encoder_hidden_size
        self.max_length = MAX_LENGTH
        self.gru_hidden_size = gru_hidden_size
        #定义embedd
        self.embedding = nn.Embedding(self.vocab_size,self.embed_size)
        #drop
        self.dropout = nn.Dropout()

        #第一个Linear层,计算注意力权重（q 与 k 拼接）
        self.attn = nn.Linear(self.embed_size+self.encoder_hidden_size,self.max_length)
        #第二个linear
        self.attn_combine = nn.Linear(self.embed_size+self.encoder_hidden_size,256)
        #gru self.gru_hidden_size = self.encoder_hidden_size
        self.gru = nn.GRU(256,self.gru_hidden_size)

        #输出层
        self.linear = nn.Linear(self.gru_hidden_size,self.vocab_size)
        #softmax
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,input,hidden,encoder_output_c):
        #input:预测的法语单词结果[1,1]
        #hidden：编码器最后单词的隐藏层张量
        #encoder：中间于一张良[10,256]

        #input--embedding[1,1,256]
        input = self.embedding(input)
        input = self.dropout(input)

        #注意力权重计算[1,10]
        self.attn_weights = F.softmax(self.attn(torch.cat((input[0],hidden[0]),dim=-1)),dim=-1)

        #计算结果与V相乘[1,1,256]
        self.attn_applied = torch.bmm(self.attn_weights.unsqueeze(0), encoder_output_c.unsqueeze(0))

        # 与Q拼接
        output1 = torch.cat((input[0], self.attn_applied[0]), dim=-1)
        output1 = self.attn_combine(output1).unsqueeze(0)

        # relu
        output1 = F.relu(output1)  # GRU模型的输入

        # 将数据送入GRU[1,1,256]
        output, hn = self.gru(output1, hidden)

        # 对output送入全连接层
        decoder_output = self.linear(output[0])
        return self.softmax(decoder_output), hn, self.attn_weights


def dm_test_AttnDecoderRNN():
    mypairsdataset = MyPairsDataset(my_pairs)#实例化数据集对象
    mydataloader = DataLoader(mypairsdataset, batch_size=1, shuffle=True)#数据加载器
    #实例化编码器
    en_vocab_size = english_word_n
    en_embed_size = 256
    en_hidden_size = 256
    my_encodergru = EncoderGRU(en_vocab_size,en_embed_size,en_hidden_size).to(device)
    #实例化解码器
    de_vocab_size = french_word_n
    de_embed_size = 256
    en_hidden_size = en_hidden_size
    gru_hidden_size = en_hidden_size
    my_attendecodergru = AttenDecoderGRU(de_vocab_size,de_embed_size,en_hidden_size,gru_hidden_size).to(device)

    #遍历数据迭代器
    for i ,(x,y) in enumerate(mydataloader):
        #一次性给数据
        hidden = my_encodergru.inithidden()
        encoder_output,encoder_hidden = my_encodergru(x,hidden)
        print('encoder_output--->',encoder_output.shape)
        print('encoder_hidden--->',encoder_hidden.shape)
        #定义中间语义张量C
        encoder_output_c = torch.zeros(MAX_LENGTH,my_encodergru.hidden_size,device=device)
        for idx in range(encoder_output.shape[1]):
            encoder_output_c[idx] = encoder_output[0][idx]
        print('encoder_output_c--->',encoder_output_c.shape)
        # for idx in range(encoder_output.shape[1]):
        #     encoder_output_c[idx] = encoder_output[0][idx]
        # print(f'encoder_output_c-->{encoder_output_c.shape}')

        #一个字符一个字符解码
        for i in range(y.shape[1]):
            temp = y[0][i].view(1,-1)
            output,hn = my_attendecodergru(temp,encoder_hidden,encoder_output_c)
            print('output--->',output.shape)
            print('hn--->',hn.shape)
            print('*'*77)
        break


mylr = 1e-4
teacher_forcing_ratio = 0.5
epochs = 10

def Train_Iters(x, y, my_encodergru, my_attndecodergru, myadam_encode, myadam_decode, mycrossentropyloss):
    #将 x 送入编码器进行处理
    hidden = my_encodergru.inithidden()
    encoed_output,encoder_hidden = my_encodergru(x,hidden)
    #语义c
    encoder_output_c = torch.zeros(MAX_LENGTH,my_encodergru.hidden_size,device=device)
    for i in range(x.shape[1]):
        encoder_output_c[i] = encoed_output[0][i]

    #
    decode_hidden = encoder_hidden

    input_y = torch.tensor([[SOS_token]],dtype=torch.long,device=device)

    #开始解码
    myloss = 0
    y_len = y.shape[1]#句子长度
    #判断 teacher_forcing
    use_teacher_forcing = True if random.random()<teacher_forcing_ratio else False
    if use_teacher_forcing:
        for idx in range(y_len):
            output, decode_hidden,atten_weights = my_attndecodergru(input_y,decode_hidden,encoder_output_c)
            target = y[0][idx].view(1)
            myloss += mycrossentropyloss(output,target)
            input_y = y[0][idx].view(1,-1)

    else:
        for idx in range(y_len):
            output, decode_hidden,atten_weights = my_attndecodergru(input_y,decode_hidden,encoder_output_c)
            target = y[0][idx].view(1)
            myloss += mycrossentropyloss(output,target)
            #根据output预测最大概率 取出
            topv,topi = output.topk(1)
            if topi.squeeze().item() == EOS_token:
                break

            input_y = topi.detach()#形状不变，为了让内存更加连续

    myadam_encode.zero_grad()
    myadam_decode.zero_grad()

    myloss.backward()

    myadam_encode.step()
    myadam_decode.step()

    return myloss.item()/y_len

















def test_TrainIter():
    #dataset
    mydataset = MyPairsDataset(my_pairs)
    print(len(mydataset))
    mydataloader = DataLoader(dataset=mydataset,batch_size=1,shuffle=True)

    #实例化编码器模型
    vocab_size = english_word_n
    embed_size = 256
    hidden_size = 256
    myencoder = EncoderGRU(vocab_size,embed_size,hidden_size).to(device)

    #attdecoder
    vocab_size = french_word_n
    embed_size = 256
    encoder_hidden_size = 256
    gru_hidden_size = 256
    mydecoder = AttenDecoderGRU(vocab_size,embed_size,encoder_hidden_size,gru_hidden_size).to(device)

    myencoder_adam = optim.Adam(myencoder.parameters(), lr=mylr)
    mydecoder_adam = optim.Adam(mydecoder.parameters(), lr=mylr)
    my_cross = nn.NLLLoss()
    for item ,(x,y) in enumerate(mydataloader):
        myloss = Train_Iters(x, y, myencoder,mydecoder,myencoder_adam,mydecoder_adam,my_cross)
        print(myloss)
        # break



#训练函数
def Train_seq2seq():
    mydataset = MyPairsDataset(my_pairs)
    mydataloader = DataLoader(mydataset,batch_size=1,shuffle=True)

    #编码器，实例化
    my_encoder = EncoderGRU(vocab_size=english_word_n,embed_size=256,hidden_size=256).to(device)

    #shilihua jiemaqi
    my_attn_decoder = AttenDecoderGRU(vocab_size=french_word_n,embed_size=256,
                                      encoder_hidden_size=256,gru_hidden_size=256).to(device)
    #实例化优化器
    my_encoder_adam = optim.Adam(my_encoder.parameters(), lr=mylr)
    my_decoder_adam = optim.Adam(my_attn_decoder.parameters(), lr=mylr)

    my_cross = nn.NLLLoss()

    #定义模型的日志参数
    plot_loss_list = []#存储平均损失，画图

    #进入循环
    for epoch_idx in range(1,1+epochs):
        #定义模型日志参数
        print_loss_total,plot_loss_total =0,0
        start_time = time.time()

        for i,(x,y) in enumerate(tqdm(mydataloader),start=1):
            #调用内置训练函数
            myloss = Train_Iters(x,y,my_encoder,my_attn_decoder,
                        my_encoder_adam,my_decoder_adam,my_cross)

            print_loss_total += myloss
            plot_loss_total += myloss

            #打印日志 每隔 1000步
            if i % 1000 == 0:
                plot_loss_avg = print_loss_total/1000
                print_loss_total = 0
                #打印日志
                end_time = time.time()
                temp_time = end_time-start_time
                print('轮次%d 损失%.6f 时间%d'% (epoch_idx,plot_loss_avg,temp_time),plot_loss_total)

            #100 步画图

            if i % 100 == 0:
                plot_loss_avg=plot_loss_total/100
                #添加列表
                plot_loss_list.append(plot_loss_avg)
                #损失归0
                plot_loss_total=0

        #每个轮次保存模型
        torch.save(my_encoder.state_dict(),'./model/ai18_encodergru_%d.pth'%epoch_idx)
        torch.save(my_attn_decoder.state_dict(),'./model/ai18_my_attn_decoder_%d.pth'%epoch_idx)

    plot_loss_list = torch.tensor(plot_loss_list).cpu()
    plt.figure()
    plt.plot(plot_loss_list)
    plt.show()
    return plot_loss_list


def seq2seq_evaluate(x,my_encoder,my_attn_decoder):
    with torch.no_grad():
        #x:预测需要翻译的英文文本
        #my_encoder:编码器模型
        #my_attn_decoder 解码器
        #
        encode_output,encode_hidden = my_encoder(x,my_encoder.inithidden())
        #jeimaqi cnashu
        #zhojain C
        encode_output_c = torch.zeros(MAX_LENGTH,my_encoder.hidden_size,device=device)
        for i in range(x.shape[1]):
            encode_output_c[i] = encode_output[0][i]

        decode_hidden = encode_hidden

        input_y = torch.tensor([[SOS_token]],dtype=torch.long,device=device)
        #自回归方式解码
        #初始化空列表存储解码的法语单词
        decoder_words = []
        #初始化一个attention的张量 获取最后每个时间步的结果  画图展示
        decoder_attentions = torch.zeros(MAX_LENGTH,MAX_LENGTH)

        #解码
        for idx in range(MAX_LENGTH):
            output,decode_hidden,attention = my_attn_decoder(input_y,decode_hidden,encode_output_c)
            #预测作为下一个时间不的输入值
            topv, topi = output.topk(1)
            #yong索引找到对应的法文单词
            predict_french_word = french_index2word[topi.item()]#一个张量可以取出来

            #存储权重
            decoder_attentions[idx] = attention

            #panduan 当前是否停止
            if topi.squeeze().item() == EOS_token:
                decoder_words.append('<EOS>')
            else:
                decoder_words.append(predict_french_word)
            input_y = topi.detach()

    return decoder_words,decoder_attentions[:idx+1]


def test_seq2seq_evaluate():
    my_encoder = EncoderGRU(vocab_size=english_word_n, embed_size=256, hidden_size=256).to(device)
    my_encoder.load_state_dict(torch.load('./model/ai18_encodergru_7.pth',map_location='cpu'))
    my_attn_decoder = AttenDecoderGRU(vocab_size=french_word_n, embed_size=256,
                                      encoder_hidden_size=256, gru_hidden_size=256).to(device)
    my_attn_decoder.load_state_dict(torch.load('./model/ai18_my_attn_decoder_7.pth',map_location='cpu'))

    my_samplepairs = [
        ['i m impressed with your french .', 'je suis impressionne par votre francais .'],
        ['i m more than a friend .', 'je suis plus qu une amie .'],
        ['she is beautiful like her mother .', 'elle est belle comme sa mere .'],
        ['i m very happy today .', 'je suis très contente aujourd hui .'],
        ['i do not want to do the school homework .', 'je ne veux pas faire les devoirs de lécole .'],
        ['i like pizza but not .', 'jaime la pizza mais pas les .'],
        ['i want to play game not learn math .', 'je veux jouer à un jeu, pas apprendre les mathématiques']
    ]

    #yuce
    for idx,pair in enumerate(my_samplepairs):
        x = pair[0]
        y = pair[1]
        #将x进行张量
        word_list = [english_word2index[word] for word in x.split(' ')]
        word_list.append(EOS_token)
        tensor_x = torch.tensor(word_list,dtype=torch.long).to(device).view(1,-1)
        decoder_words,attentions = seq2seq_evaluate(x =tensor_x,my_encoder=my_encoder,
                                                    my_attn_decoder=my_attn_decoder)

        decode_french = ' '.join(decoder_words)
        print('x原始输入的英文句子',x)
        print('真实结果',y)
        print('模型预测结果', decode_french)
        print('*'*44)








def test_attention():
    my_encoder = EncoderGRU(vocab_size=english_word_n, embed_size=256, hidden_size=256).to(device)
    my_encoder.load_state_dict(torch.load('./model/ai18_encodergru_10.pth',map_location='cpu'))
    my_attn_decoder = AttenDecoderGRU(vocab_size=french_word_n, embed_size=256,
                                      encoder_hidden_size=256, gru_hidden_size=256).to(device)
    my_attn_decoder.load_state_dict(torch.load('./model/ai18_my_attn_decoder_10.pth',map_location='cpu'))


    sentence = 'we are both teacher'
    #yuce

    #将x进行张量
    word_list = [english_word2index[word] for word in sentence.split(' ')]
    word_list.append(EOS_token)
    tensor_x = torch.tensor(word_list,dtype=torch.long).to(device).view(1,-1)
    decoder_words,attentions = seq2seq_evaluate(x =tensor_x,my_encoder=my_encoder,
                                                my_attn_decoder=my_attn_decoder)

    decode_french = ' '.join(decoder_words)
    print('x原始输入的英文句子',sentence)
    print('模型预测结果', decode_french)
    print('*'*44)

    plt.matshow(attentions.numpy())
    plt.show()








if __name__ == '__main__':
    english_word2index,english_index2word,english_word_n,french_word2index,french_index2word,french_word_n,my_pairs = my_getdata()
    # print(len(english_word2index))
    # a = MyPairsDataset(my_pairs)
    # print(len(a))
    # dm_test_MyPairsFataset()
    # dm_test_EncoderGRU()
    # dm03_test_DecoderGRU()
    # dm_test_AttnDecoderRNN()
    # test_TrainIter()
    # test_TrainIter()
    # Train_seq2seq()
    # test_seq2seq_evaluate()
    # test_attention()

