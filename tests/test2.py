import sys
sys.path.append("./freechat")
sys.path.append("./registrator")
sys.path.append("./reg_model")
sys.path.append("./classificator")

import freechat as fch
import registrator as reg
import classificator as cls

if(__name__ == "__main__"):
    chatbot = fch.FreeChatBot()
    registrator = reg.Registrator("./reg_model/vocab.pt")
    
    replics1 = ["Хорошая сегодня погода",
               "Какие книги вам нравятся?",
               "У вас много туристов?",
               "Давно были в кино?"]
    
    replics2 = ["Завтра побегу марафон",
               "Сегодня дождь целый день",
               "Красивая ручка",
               "Мне нравятся итальянские машины"]
    
    replics3 = ["Я люблю фантастику, особенно Лукъяненко",
               "Кто написал эту картину?",
               "Где можно провести время вечером?"]
    
    for replic in replics1:
        print(f"Пользователь >> {replic}")
        if (replic == "exit"):
            break
        if(cls.classify(replic)):
            print(f"Регистратор >> {registrator.generateRegistratorAnswer(replic)}")
        else:
            print(f"Регистратор >> {chatbot.generateFreeAnswer(replic)}")
            
print("\n\n\n")
for replic in replics2:
        print(f"Пользователь >> {replic}")
        if (replic == "exit"):
            break
        if(cls.classify(replic)):
            print(f"Регистратор >> {registrator.generateRegistratorAnswer(replic)}")
        else:
            print(f"Регистратор >> {chatbot.generateFreeAnswer(replic)}")
        
print("\n\n\n")
for replic in replics3:
        print(f"Пользователь >> {replic}")
        if (replic == "exit"):
            break
        if(cls.classify(replic)):
            print(f"Регистратор >> {registrator.generateRegistratorAnswer(replic)}")
        else:
            print(f"Регистратор >> {chatbot.generateFreeAnswer(replic)}")