import re
from progress.bar import ChargingBar 
import time

data_path = "../Data/"
lines = []

def find_name(text):
    text = str(text)
    text.strip()
    pattern = r'((\b[A-Я][^A-Я\s\.\,][a-я]*)(\s+)([A-Я][a-я]*)'+\
    '(\.+\s*|\s+)([A-Я][a-я]*))'
    
    name = re.findall(pattern, text)
    if name:
        return True
    else: 
        return False

def classify(input_text):
    
    if find_name(input_text):
        return True
    input_text = input_text.lower().rstrip().lstrip()
    reg_word = ["извините", "прощения",
                "добрый веч", "здравствуйте", "приве",
                "нужен номер", "места", "остановиться", "номер",
                "бронь на меня", "бронь на меня", "номер будет", "номер брони",
                "будет забронирован", "бронь на меня", "бронь будет",
                "да, пожалуйста", "не моя вина", "можно, побыстрее",
                "одна ночь", "одну ночь", "номер на одного", "только я",
                "никого нет", "я постою", "хорошо", "побыстрее", "я устал",
                "нет проблем", "спешу", "подешевле", "большой", "маленький",
                "средний", "отличное", "день был", "неплохо" 
                ]
    count = 0
    for i in reg_word:
        if (input_text.find(i) >= 0):
            count += 1
    if (count > 0):
        return True
    else:
        return False 

