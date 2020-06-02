#导入库
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import  pyplot as plt
from matplotlib import ticker
from nltk.translate.bleu_score import sentence_bleu
import time,random,os,jieba,logging
import numpy as np
import pandas as pd
jieba.setLogLevel(logging.INFO)

#定义开始符和结束符
sosToken = 1
eosToken = 0

#定义Encoder
class EncoderRNN(nn.Module):
    #初始化
      def __init__(self,featureSize,hiddenSize,embedding,numLayers=1,dropout=0.1,bidirectional=True):
          super(EncoderRNN, self).__init__()
          self.embedding = embedding
          #核心API
          self.gru = nn.GRU(featureSize,hiddenSize,num_layers=numLayers,dropout=(0 if numLayers==1 else dropout),bidirectional =bidirectional,batch_first=True)
          #超参
          self.featureSize = featureSize
          self.hiddenSize = hiddenSize
          self.numLayers = numLayers
          self.bidirectional = bidirectional

          #前向计算，训练和测试中必须的部分
      def forward(self,input,lengths,hidden):
          #input: batchSize*seq_len ; hidden: numLayers*d*batchSize*hiddenSize
          #给定输入
          input = self.embedding(input)   #batchSize*seq_len*feaSize
          #加入paddle 方便计算
          packed = nn.utils.rnn.pack_padded_sequence(input,lengths,batch_first=True)
          output,hn = self.gru(packed,hidden) #output: batchSize*seq_len*hiddenSize*d  hn:numLayers*d*batchSize*hiddenSize
          output,__ = nn.utils.rnn.pack_padded_sequence(output,batch_first=True)
          #判断是否使用双向GRU
          if self.bidirectional:
              output = output[:,:,:self.hiddenSize]+output[:,:,:self.hiddenSize]
          return output,hn
#定义Decoder
class DecoderRNN(nn.Module):
    #初始化
    def __init__(self,featureSize,hiddenSize,outputSize,embedding,numLayers=1,dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.embedding = embedding
        #核心API 搭建GRU
        self.gru = nn.GRU(featureSize,hiddenSize,num_layers=numLayers,batch_first=True)
        self.out = nn.Linear(featureSize,outputSize)

        #定义前向计算
    def forward(self,input,hidden):
        input = self.embedding(input)
        #relu 激活 softmax计算结果
        input = F.relu(input)
        output,hn = self.gru(input,hidden)
        output = F.log_softmax(self.out(output),dim=2)#output: batchSize*seq_len*outputSize
        return output,hn,torch.zeros([input.size(0),1,input.size(1)])
#定义B Attention的Decoder
class BahdanauAttentionDecoderRNN(nn.Module):
    #初始化
    def __init__(self,featureSize,hiddenSize,outputSize,embedding,numLayers=1,dropout=0.1):
        super(BahdanauAttentionDecoderRNN, self).__init__()
        self.embedding = embedding

        #定义attention的权重，如何去联合，以及dropout
        self.dropout = nn.Dropout(dropout)
        self.attention_weight = nn.Linear(hiddenSize*2,1)
        self.attention_combine = nn.Linear(featureSize+hiddenSize,featureSize)
        #核心API 搭建GRU层，并给定超参数
        self.gru = nn.GRU(featureSize,hiddenSize,num_layers=numLayers,dropout=(0 if numLayers==1 else dropout),batch_first=True)
        self.out = nn.Linear(hiddenSize,outputSize)
        self.numLayers = numLayers

        #定义前向计算
    def forward(self,inputStep,hidden,encoderOutput):
        #防止过拟合
          inputStep = self.embedding(inputStep)
          inputStep = self.dropout(inputStep)
        #计算attetion的权重 本质Softmax
          attentionWeight = F.softmax(self.attention_weight(torch.cat((encoderOutput,hidden[-1:].expand(encoderOutput.size(1),-1,-1).transpose(0,1)),dim=2)).transpose(1,2),dim=2)
          context = torch.bmm(attentionWeight,encoderOutput)#context:batchSIze*1*hiddenSize
          attentionCombine = self.attention_combine(torch.cat((inputStep,context),dim=2)) # combine:batchSIze*feaSize
          attentionInput = F.relu(attentionCombine)
          output,hidden = self.gru(attentionInput,hidden)
          output = F.softmax(self.out(output),dim=2)
          return output,hidden,attentionWeight

#定义L attention
class LuongAttention(nn.Module):
    #初始化
     def __init__(self,method,hiddenSize):
         super(LuongAttention, self).__init__()
         self.method = method
         #三种模式 dot  general concat
         if self.method not in ['dot','general','concat']:
             raise  ValueError(self.method,"is not an attetion method.")
         if self.method=='general':
             self.Wa = nn.Linear(hiddenSize,hiddenSize)
         if self.method=='concat':
             self.Wa = nn.Linear(hiddenSize*2,hiddenSize)
             self.v = nn.Parameter(torch.FloatTensor(1,hiddenSize))
     #给出dot计算方法
     def dot_score(self,hidden,encoderOutput):
         return torch.sum(hidden*encoderOutput,dim=2)
     #给出general的计算方法
     def general_score(self,hidden,encoderOutput):
        energy = self.Wa(encoderOutput)
        return torch.sum(hidden*energy,dim=2)
    #给出concat的计算法
     def concat_score(self,hidden,encoderOutput):
        energy = torch.tanh(self.Wa(torch.cat((hidden.expand(-1,encoderOutput(1),-1),encoderOutput),dim=2)))
        return torch.sum(hidden * energy, dim=2)

     #定义前向计算
    def forward(self,hidden,encoderOutput):
        #3选1
         if self.method == 'general':
             attentionScore = self.general_score(hidden,encoderOutput)
         elif self.method == 'concat':
             attentionScore = self.concat_score(hidden,encoderOutput)
         elif self.method == 'dot':
             attentionScore = self.dot_score(hidden,encoderOutput)
         return F.softmax(attentionScore,dim=1)

#定义L Attention Decoder
class LuongAttentionDecoderRNN(nn.Module):
    #初始化
    def __init__(self,featureSize,hiddenSize,outputSize,embedding,numLayers=1,dropout=0.1,attnMethod='dot'):
        super(LuongAttentionDecoderRNN, self).__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        #核心API 搭建GRU
        self.gru = nn.GRU(featureSize,hiddenSize,num_layers=numLayers,dropout=(0 if numLayers==1 else dropout),batch_first=True)
        #定义权重计算和连接的方式
        self.attention_weight = LuongAttention(attnMethod,hiddenSize)
        self.attention_combine = nn.Linear(hiddenSize*2,hiddenSize)
        self.out = nn.Linear(hiddenSize,hiddenSize)
        self.numLayers = numLayers

    #定义前向计算
    def forward(self,inputStep,hidden,,encoderOutput):
        #对输入进行dropout
        inputStep = self.embedding(inputStep)
        inputStep = self.dropout(inputStep)
        #对输出计算
        output,hidden = self.gru(inputStep,hidden)
        #attention 权重计算
        attentionWeight = self.attetion_weight(output,encoderOutput)
        context = torch.bmm(attentionWeight,encoderOutput)
        attentionCombine = self.attention_combine(torch.cat(output,context),dim=2)
        attentionOutput = torch.tanh(attentionCombine)
        #最终的output
        output = F.log_softmax(self.out(attentionOutput),dim=2)
        return output,hidden,attentionWeight

 #如何去选择decoder   L 、B  、None
 def __DecoderRNN(attnType,featureSize,hiddenSize,outputSize,embedding,numLayers,dropout,attnMethod):
     #使用哪个attention
       if attnType not in ['L','B','None']:
           raise ValueError(attnType,"is not an appropriate attention type")
       if attnType == 'L':
           return LuongAttentionDecoderRNN(featureSize,hiddenSize,outputSize,embedding=embedding,numLayers=numLayers,dropout=dropout,attnMethod=attnMethod)
       elif attnType == 'B':
           return BahdanauAttentionDecoderRNN(featureSize,hiddenSize,outputSize,embedding=embedding,numLayers=numLayers,dropout=dropout)
       else:
           return DecoderRNN(featureSize,hiddenSize,outputSize,embedding=embedding,numLayers=numLayers,dropout=dropout)
#seq2seq 定义核心类
class seq2seq:
    #初始化
        def __init__(self,dataClass,featureSize,hiddenSize,encoderNumLayers=1,decoderNumLayers=1,attnType='L',attnMethod='dot',dropout=0.1,encoderBidirectional=False,outputSize=None,embedding=None,device=torch.device("cpu")):
            #定义输出的维度
            outputSize = outputSize if outputSize else dataClass.wordNum
            embedding = embedding if embedding else nn.Embedding(outputSize+1,featureSize)
            #数据读入
            self.dataClass = dataClass
            #搭建模型架构
            self.featureSize = featureSize
            self.hiddenSize = hiddenSize
            #encoder调用 构建
            self.encoderRNN = EncoderRNN(featureSize,hiddenSize,embedding=embedding,numLayers=encoderNumLayers,dropout=dropout,bidirectional=encoderBidirectional).to(device)
            #decoder 构建
            self.decoderRNN = __DecoderRNN(attnType,featureSize,hiddenSize,outputSize,embedding=embedding,numLayers=decoderNumLayers,dropout=dropout,attnMethod=attnMethod).to(device)
            self.embedding = embedding.to(device)
            self.device = device

        #定义训练方法：
        def train(self,batchSize,isDataEnhance=False,dataEnhanceRatio=0.2,epoch=100,stopRound=10,lr=0.001,betas=(0.9,0.99),eps=le-08,weight_decay=0,teacherForcingRatio=0.5):

              #使用哪个api训练
              self.encoderRNN.train(),self.decoderRNN.train()
             #给定batchSize 是否使用数据增广
              batchSize = min(batchSize,self.dataClass.trainSampleNum) if batchSize>0 else self.dataClass.trainSampleNum
              dataStream = self.dataClass.random_batch_data_stream(batchSize=batchSize,isDataEnhance=isDataEnhance,dataEnhanceRatio=dataEnhanceRatio)

              #定义优化器 Adam
            # 对于测试数据batch制作
               if self.dataClass.testsize>0:testStrem=self.dataClass.random_batch_data_stream(batchSize=batchSize,type="test")
               itersPerEpoch =self.dataClass.trainSampleNum//batchSize
               encoderOptimzer = torch.optim.Adam(self.encoderRNN.parameters(),lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
               decoderOptimzer = torch.optim.Adam(self.decoderRNN.parameters(),lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
               st = time.time()
             #做循环
               for e in range(epoch):
                    for i in range(itersPerEpoch):
                        X,XLens,Y,YLens = next(dataStream)
                        loss = self. _train_step(X,XLens,Y,YLens,encoderOptimzer,decoderOptimzer,teacherForcingRatio)
             # bleu embAve
                        if (e*itersPerEpoch+i+1)%stopRound==0:
                            bleu = _bleu_score(self.encoderRNN,self.decoderRNN,X,XLens,Y,YLens,self.dataClass.maxSentLen,device=self.device)
                            embAve = _embAve_score(self.encoderRNN,self.decoderRNN,X,XLens,Y,YLens,self.dataClass.maxSentLen,device=self.device)
                            print("After iters%d:loss =  %.3lf;train bleu:%.3lf,embAve:%.3lf"%(e*itersPerEpoch+i+1,loss,bleu,embAve),end='')
                        resetNum = ((itersPerEpoch-i-1)+(epoch-e-1)*itersPerEpoch)*batchSize
                        speed = ((e*itersPerEpoch+i+1) *batchSize//(time.time()-st)
                        print ("%.3lf qa/s; remaining time:%.3lf"%(speed,resetNum/speed))

                        #保存模型
                        def save(self,path):
                            torch.save({"encoder":self.encoderRNN,"decoder":self.decoderRNN,
                                               "word2id":self.dataClass.word2id,"id2word":self.dataClass.id2word},path)
                            print('Model saved in "%s".'%path)

                        #训练中的梯度计算及loss计算 Bp
                        def _train_step(self,X,XLens,Y,YLens,encoderOptimzer,decoderOptimzer,teacherForcingRatio):
                            #计算梯度 实现BP
                             encoderOptimzer.zero_grad()
                             decoderOptimzer.zero_grad()
                            #计算loss
                             loss,nTotal = _calculate_loss(self.encoderRNN,self.decoderRNN,X,XLens,Y,YLens,teacherForcingRatio,device=self.device)

                            #实现BP
                             (loss/nTotal).backward()
                             encoderOptimzer.step()
                             decoderOptimzer.step()
                             return loss.item() / nTotal

        #读入数据预处理 进行操作
from model.pre_process import seq2id,id2seq,filter_sent
class ChatBot:
    def __init__(self,modelPath,device=torch.device('cpu')):
        modelDict = torch.load(modelPath)
        self.encoderRNN,self.decoderRNN = modelDict['encoder'].to(device),modelDict['decoder'].to(device)
        self.word2id,self.id2word = modelDict['word2id'],modelDict['id2word']
        self.hiddenSize = self.encoderRNN.hiddenSize
        self.device = device

        #验证模型
        self.encoderRNN.eval(),self.decoderRNN.eval()

     #定义贪婪搜索，inference 中使用
    def predictByGreedySearch(self,inputSeq,maxAnswerLength=32,showAttention=False,figsize=(12,6))):
        inputSeq = filter_sent(inputSeq)
        inputSeq = [w for w in jieba.lcut(inputSeq) if w in self.word2id.keys()] #fenci
        X = seq2id(self.word2id,inputSeq)
        XLens = torch.tensor([len(X)+1],dtype=torch.int,device=self.device)
        X = X +[eosToken]  #加终止符
        X = torch.tensor([X],dtype=torch.long,device=self.device)
    #定义相关的层，并且确定相应的encoder确定隐层
        d = int(self.encoderRNN.bidirectional)+1
        hidden = torch.zeros((d*self.encoderRNN.numLayers,1,self.hiddenSize(),dtype=torch.float32,device=self.device))
        encoderOutput,hidden = self.encoderRNN(X,XLens,hidden)
        hidden = hidden[-d*self.decoderRNN.numLayers::2].contiguous()

        attentionArrs = []
        Y = []
        #定义decoder输入
        decoderInput = torch.tensor([[sosToken]],dtype=torch.long,device=self.device)
        #写循环
        while decoderInput.item() != eosToken and len(Y)<maxAnswerLength:
                decoderOutput,hidden,decoderAttentionWeight =self.decoderRNN(decoderInput,hidden,encoderOutput)
                topv,topi = decoderOutput.topk(1)
                decoderInput = topi[: , : , 0]
                attentionArrs.append(decoderAttentionWeight.data.cpu().numpy().reshape(1,XLens))
                Y.append(decoderInput.item() )
         outputSeq = id2seq(self.id2word,Y)

        #是否可视化attention
        if showAttention:
            attentionArrs= np.vstack(attentionArrs)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot('111')
            cax = ax.matshow(attentionArrs,cmap='bone')
            fig.colorbar(cax)
            ax.set_yticklabels(['','<SOS>']+inputSeq)
            ax.set_yticklabels(['',]+outputSeq)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            plt.show()
        return ''.join(outputSeq[:-1])


#beamSearch
def predictByBeamSearch(self,inputSeq,beamWidth=10,maxAnswerLength=32,alpha=0.7,isRandomChoosse=False,allRandomChoose=False,improve=True,showInfo=False):
    #定义输出
       outputSize = len(self.id2word)
       inputSeq = filter_sent(inputSeq)
       inputSeq = [w for w in jieba.lcut(inputSeq) if w in self.word2id.keys()]  # fenci
       X = seq2id(self.word2id, inputSeq)
       XLens = torch.tensor([len(X) + 1], dtype=torch.int, device=self.device)
       X = X + [eosToken]  # 加终止符
       X = torch.tensor([X], dtype=torch.long, device=self.device)
    #使用双向GRUencoder和2层GRUdecoder
       d = int(self.encoderRNN.bidirectional)+1
       hidden = torch.zeros((d*self.encoderRNN.numLayers,1,self.hiddenSize(),dtype=torch.float32,device=self.device)
       encoderOutput, hidden = self.encoderRNN(X, XLens, hidden)
       hidden = hidden[-d * self.decoderRNN.numLayers::2].contiguous()

       #把搜索宽带和最大回答长度做一些处理
       Y = np.ones([beamWidth,maxAnswerLength],dtype='int32')*eosToken
       #prob :beamWidth*1
       prob =np.zeros([beamWidth,1],dtype='float32')
    # 定义decoder输入
       decoderInput = torch.tensor([[sosToken]], dtype=torch.long, device=self.device)
       #定义decoder输出
       decoderOutput, hidden, decoderAttentionWeight = self.decoderRNN(decoderInput, hidden, encoderOutput)
       #topv 1*1*beamWidth  topi
       topv,topi = decoderOutput.topk(beamWidth)
       decoderInput = topi.view(beamWidth,1)
       #做循环
       for i in range(beamWidth):
           Y[i,0] = decoderInput[i].item()
        Y_ = Y.copy()
        #prob
        prob + = topv.view(beamWidth,1).data.cpu().numpy()
        prob_ = prob.copy()
        #hidden
        hidden = hidden.expand(-1,beamWidth,1).contiguous()
        localRestId = np.array([i for i in range(beamWidth)],dtype='int32')
        encoderOutput = encoderOutput.expand(beamWidth,-1,-1)
        #计算权重 包括improve allRandomChoose
        for i in range(1,maxAnswerLength):
            decoderOutput, hidden, decoderAttentionWeight = self.decoderRNN(decoderInput, hidden, encoderOutput)
            #improve
            if improve:
                decoderOutput = decoderOutput.view(-1,1)
                #allRandomChoose
                if allRandomChoose:
                    topv,topi = self._random_pick_k_by_prob(decoderOutput,k=beamWidth)
                else:
                    topv,topi = decoderOutput.topk(beamWidth,dim=0)

                decoderInput = topi%outputSize
                #计算部分 一些逻辑，同时去找到结果答案
                idFrom = topi.cpu().view(-1).numpy()//outputSize
                Y[localRestId,:i+1] = np.hstack([Y[localRestId[idFrom],:i],decoderInput.cpu().numpy()])
                prob[localRestId] = prob[localRestId[idFrom]]+topv.data.cpu().numpy()
                hidden = hidden [:idFrom:]

                #restId
                restId =(decoderInput!=eosToken).cpu().view(-1)
                localRestId = localRestId[restId.numpy().astype('bool')]
                decoderInput = decoderInput[restId]
                hidden =  hidden [:restId:]
                encoderOutput = encoderOutput(restId)
                beamWidth = len(localRestId)
                #搜索直到0为止
                if beamWidth<1:
                    break
            #lens
            lens = [i.index(eosToken)if eosToken in i else maxAnswerLength for i in Y.tolist()]
            ans = [''.join(id2seq(self.id2word,i[:l]))for i,l in zip(Y,lens)]
            prob = [prob[i,0]/np.power(lens[i],alpha)for i in range(len(ans))]

            #给定参数的策略
            if isRandomChoosse or allRandomChoose:
                prob = [np.exp(p) for p in prob]
                prob = [np.sum(prob) for p in prob]
                if showInfo:
                    for i in range(len(ans))):
                        print((ans[i],prob[i]))
                    return random_pick(ans,prob)
                else:
                    ansAndProb = list(zip(ans,prob))
                    ansAndProb.sort(key=lambda x:x[1],reverse=True)
                    if showInfo:
                        for i in ansAndProb:
                            print(i)
                        return ansAndProb[0][0]
# 定义验证方法
       def evaluate(self,dataClass,batchSize=128,isDataEnhance=False,dataEnhanceRatio=0.2,streamType='train'):
           #数据处理
             dataClass.reset_word_id_map(self.id2word,self.word2id)
             dataStream = dataClass.one_epoch_data_stream(batchSize=batchSize,isDataEnhance=isDataEnhance,dataEnhanceRatio=dataEnhanceRatio,type=streamType)
             bleuScore,embAveScore = 0.0,0.0
             totalSampleNum = dataClass.trainSampleNum if streamType=='train' else dataClass.testSampleNum
             iters = 0
             st = time.time()
             while True:
                 try:
                     X,XLens,Y,YLens = next(dataStream)
                 except:
                     break
                #bleu 和embAve的计算
                 bleuScore += _bleu_score(self.encoderRNN,self.decoderRNN,X,XLens,Y,YLens,dataClass.maxSentLen,self.device,mean=False)
                 embAveScore += _embAve_score(self.encoderRNN,self.decoderRNN,X,XLens,Y,YLens,dataClass.maxSentLen,self.device,mean=False)
                 iters += len(X)
                 finishedRatio = iters/totalSampleNum
                 print('Finised %. 3lf%%; remaining time : %.3lfs' %(finishedRatio*100.0,(time,time()-st)*(1.0-finishedRatio/finishedRatio))
             return bleuScore/totalSampleNum,embAveScore/totalSampleNum

       #根据概率随机取K个结果
       def _random_pick_k_by_prob(self,decoderOutput,k):
           df = pd.DataFrame([[i] for i in range(len(decoderOutput))])
           prob = torch.softmax(decoderOutput.data,dim=0).cpu.numpy().reshape(-1)
           topi = torch.tensor(np.array(df.sample(n=k,weights=prob)),dtype=torch.long,device=self.device)
           return  decoderOutput[topi.view(-1)],topi

       #随机选一个prob比较大的
def random_pick(sample,prob):
    x = random.uniform(0,1)
    cntProb = 0.0
    for sampleItem,probItem in zip(sample,prob):
        cntProb += probItem
        if x<cntProb:break
    return  sampleItem

#bleu的评价指标，机器翻译的指标
def _bleu_score(encoderRNN,decoderRNN,X,XLens,Y,YLens,maxSentLen,device,mean=True):
    Y_pre = _calculate_Y_pre(encoderRNN,decoderRNN,X,XLens,Y,YLens,maxSentLen,techerForcingRatio=0,device=device)
    Y = [list(Y[i])[:YLens[i]-1]for i in range (len(YLens))]
    Y_pre = Y_pre.cpu().data.numpy()
    Y_preLens = [list(i).index(0)if 0 in i else len(i) for i in Y_pre]
    bleuScore = [sentence_bleu([i],j,weights=(1,0,0,0))for i,j in zip(Y,Y_pre)]
    return np.mean(bleuScore) if mean else np.sum(bleuScore)

#embAve的评价指标，类似平方差之类的
def _embAve_score(encoderRNN,decoderRNN,X,XLens,Y,YLens,maxSentLen,device,mean=True):
    Y_pre = _calculate_Y_pre(encoderRNN, decoderRNN, X, XLens, Y, YLens, maxSentLen, techerForcingRatio=0,
                             device=device)
    Y_pre = Y_pre.data
    Y_preLens = [list(i).index(0) if 0 in i else len(i) for i in Y_pre]
    #emb?
    emb = encoderRNN.embedding
    Y,Y_pre = emb(torch.tensor(Y,dtype=torch.long,device=device)).cpu().data.numpy(),emb(Y_pre).cpu().data.numpy()

    #sentence Vector 定义句子的向量和预测句子的向量
    sentVec = np.array([np.mean(Y[i:YLens[i]],axis=0)for i in range(len(Y))],dtype='float32')
    sent_preVec = np.array([np.mean(Y_pre[i:Y_preLens[i]],axis=0)for i in range(len(Y_pre))],dtype='float32')
    #计算得分
    embAve_Score = np.sum(sentVec*sent_preVec,axis=1)/(np.sqrt(np.sum(np.square(sentVec),axis=1)))
    return np.mean( embAve_Score) if mean else np.sum(embAve_Score)

#计算loss
def _calculate_loss(encoderRNN,decoderRNN,X,XLens,Y,YLens,teacherForcingRatio,device):
    featureSize,hiddenSize =encoderRNN.featureSize,encoderRNN.hiddenSize
    X,Y = torch.tensor (X,dtype=torch.long,device=device),torch.tensor (Y,dtype=torch.long,device=device)
    XLens,YLens = torch.tensor(XLens,dtype=torch.int,device=device),torch.tensor (YLens,dtype=torch.int,device=device)

    #定义batchsize
    batchSize = X.size(0)
    XSeqLen,YSeqLen = X.size(1),YLens.max().item()
    encoderOutput = torch.zeros((batchSize,XSeqLen,featureSize),dtype='float32',device=device)

    #d
    d = int(encoderRNN.bidirectional)+1
    hidden = torch.zeros((d*encoderRNN.numLayers,batchSize,hiddenSize),dtype=torch.float32,device=device)
    #sort
    XLens,indices = torch.sort(XLens,descending=True)
    _,desortedIndices = torch.sort(indices,descending=False)
    encoderOutput,hidden = encoderRNN(X[indices],XLens,hidden)
    encoderOutput,hidden = encoderOutput[desortedIndices],hidden[-d*decoderRNN.numLayers::d,desortedIndices,:]
    #定义decoder输入
    decoderInput = torch.tensor([[sosToken]for i in range(batchSize)],dtype=torch.long,device=device)
    loss,nTotal = 0,0
    #循环，遍历，对于每个decoder过程中都会取top并计算loss
    for i in range(YSeqLen):
        decoderOutput,hidden,decoderAttentionWeight =decoderRNN(decoderInput,hidden,encoderOutput)
        loss += F.nll_loss(decoderOutput[:0:],Y[:,i],reduction='sum')
        nTotal += len(decoderInput)
        #teacherForcingRatio
        if random.random()<teacherForcingRatio:
            decoderInput = Y[:,i:i+1]
        else:
            topv,topi = decoderOutput.topk(1)
            decoderInput = topi [:,:,0]
         #restid
        restId = (YLens>i+1).view(-1)
        decoderInput = decoderInput(restId)
        hidden = hidden[:,restId,:]
        encoderOutput = encoderOutput[restId]
        Y = Y[restId]
        YLens = YLens[restId]
    return loss,nTotal

#计算Y_pre的方法
def _calculate_Y_pre(encoderRNN,decoderRNN,X,XLens,Y,YMaxLen,teacherForcingRatio,device):
    featureSize,hiddenSize =encoderRNN.featureSize,encoderRNN.hiddenSize
    X,Y = torch.tensor (X,dtype=torch.long,device=device),torch.tensor (Y,dtype=torch.long,device=device)
    XLens = torch.tensor(XLens,dtype=torch.int,device=device)

    # 定义batchsize
    batchSize = X.size(0)
    XSeqLen = X.size(1)
    encoderOutput = torch.zeros((batchSize, XSeqLen, featureSize), dtype='float32', device=device)

    #d
    d = int(encoderRNN.bidirectional)+1
    hidden = torch.zeros((d*encoderRNN.numLayers,batchSize,hiddenSize),dtype=torch.float32,device=device)
    #sort
    XLens,indices = torch.sort(XLens,descending=True)
    _,desortedIndices = torch.sort(indices,descending=False)
    encoderOutput,hidden = encoderRNN(X[indices],XLens,hidden)
    encoderOutput,hidden = encoderOutput[desortedIndices],hidden[-d*decoderRNN.numLayers::d,desortedIndices,:]
    # 定义decoder输入
    decoderInput = torch.tensor([[sosToken] for i in range(batchSize)], dtype=torch.long, device=device)
    Y_pre,localRestId = torch.ones([batchSize,YMaxLen],dtype=torch.long,device=device)*eosToken,torch.tensor([i for i in range(batchSize)],dtype=torch.long,device=device)
    #做循环
    for i in range(YMaxLen):
        decoderOutput,hidden,decoderAttentionWeight =decoderRNN(decoderInput,hidden,encoderOutput)
        # teacherForcingRatio
        if random.random() < teacherForcingRatio:
            decoderInput = Y[:, i:i + 1]
        else:
            topv,topi = decoderOutput.topk(1)
            decoderInput = topi [:,:,0]

        #Y_pre
        Y_pre[localRestId,i] = decoderInput.squeeze()
        restId = (decoderInput!=eosToken).view(-1)
        localRestId = localRestId(restId)
        decoderInput = decoderInput(restId)
        hidden = hidden[:,restId:]
        encoderOutput = encoderOutput[restId]
        Y = Y [restId]
        if len(localRestId)<1:
            break
    return Y_pre





