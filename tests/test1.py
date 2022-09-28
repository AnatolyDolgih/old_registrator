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
    
    replics = ["Извините, я тороплюсь.",
               "Добрый вечер.",
               "Мне нужен номер.",
               "Мне сказали, что номер будет забронирован, но номер не сообщили.",
               "Долгих Анатолий Андреевич",
               "Возможно это ошибка? Мне сказали, что бронь на меня есть.",
               "Это не моя вина: постарайтесь уладить.",
               "Всего на одну ночь.",
               "Только я.",
               "Нет проблем.",
               "Да, я сильно спешу.",
               "Хотелось бы подешевле.",
               "большой",
               "Отличное!",
               "Да, хорошо.",
               "День был прекрасный.",
               "Здесь неплохо.",
               "Хорошо, давайте продолжим оформление.",
               "Нет, я устал(а) иду спать."]
    
    for replic in replics:
        print(f"Пользователь >> {replic}")
        if (replic == "exit"):
            break
        if(cls.classify(replic)):
            print(f"Регистратор >> {registrator.generateRegistratorAnswer(replic)}")
        else:
            print(f"Регистратор >> {chatbot.generateFreeAnswer(replic)}")
    
