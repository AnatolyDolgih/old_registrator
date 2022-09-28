import sys
sys.path.append("./freechat")
sys.path.append("./registrator")
sys.path.append("./reg_model")
#sys.path.append("./classificator")

import freechat as fch
import registrator as reg
#import classificator as cls
from classificator import classificator as cls
from classificator import filter as flt

if(__name__ == "__main__"):
    chatbot = fch.FreeChatBot("./freechat/RuDialoGPT/model")
    registrator = reg.Registrator("./reg_model/vocab.pt", "./reg_model/saved_model/model_params_20_epoch.pth")
    while True:
        replic = input('Пользователь >> ')
        if (replic == "exit"):
            break
        if(cls.classify(replic)):
            print(f"Регистратор >> {registrator.generateRegistratorAnswer(replic)}")
        else:
            check = True
            while (check == True):
                answer = chatbot.generateFreeAnswer(replic)
                if flt.filter(answer):
                    print(f"Регистратор >> {answer}")
                    check = False    
                
    
