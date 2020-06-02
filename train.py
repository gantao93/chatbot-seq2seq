#导入库
import torch
from model.model import *
from model.pre_process import *

#读入数据
dataclass =  Corpus('./data/qingyun.tsv',maxSentenceWordsNum=25)

#指定模型和一些超参数
model = Seq2Seq(dataclass,featureSize=256,hiddenSize=256,attnType='L',
                attnMethod='general',encoderNumLayers=5,decoderNumLayers=3,encoderBidirectional=True,device=torch.device('cuda:0)'))
#训练
model.train(batchSize=1024,epoch=500)

#保存模型
model.save('model.pkl')

