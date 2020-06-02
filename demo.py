#导入库
import torch,warnings,argparse
from model.model import ChatBot
warnings.filterwarnings("ignore")


#选择哪个模型，是否使用GPU
parser = argparse.ArgumentParser()
parser.add_argument ('--model',help='The path of your model file',required=True,type=str)
parser.add_argument ('--device',help='your program running environment,"cpu"or "cuda"',type=str,default='cpu')
args = parser.parse_args()
print(args)

#主程序，打印log，显示对话
if __name__  == "__main__":
    print('Loading the model...')
    ChatBot = ChatBot(args.model,device=torch.device(args.device))
    print('finished...')

    allRandomChoose,showInfo= False,False

    #再终端要显示对话
    while True:
        inputSeq = input ("主人：")
        if inputSeq=='_crazy_on':
            allRandomChoose=True
            print('小可爱：','成功开启疯狂模式...')
        elif inputSeq=='_crazy_off':
            allRandomChoose = False
            print('小可爱：', '成功关闭疯狂模式...')
        elif inputSeq == '_showInfo_on_':
            showInfo = True
            print('小可爱：', '成功开启日志...')
        elif inputSeq == '_showInfo_off_':
            showInfo = False
            print('小可爱：', '成功关闭日志...')
        else:
            outputSeq = ChatBot.predictByBeamSearch(inputSeq,isRandomChoose=True,allRandomChoose=allRandomChoose,showInfo=showInfo)
            print('小可爱：',outputSeq)
        print()
