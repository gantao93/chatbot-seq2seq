#导入库
import torch
from model.model import *
from  model.pre_process import *

#载入模型
ChatBot = ChatBot('model.pkl')

#针对一些问题给出回复内容
ChatBot.predictByGreedySearch("你好啊")

#针对一些其他的问题给出回复内容
ChatBot.predictByBeamSearch("什么是人工智能",isRandomChoose=True,beamWidth=10)
