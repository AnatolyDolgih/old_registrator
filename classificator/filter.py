Bad_list_word = ["бля", "жоп", "еба", "ху", "пидо", "пизд", 
                 "ганд", "долб", "сц", "дроч"]

def filter(input_text):
    input_text = input_text.lower()
    check = 0
    for word in Bad_list_word:
        if (input_text.find(word) >= 0):
            check += 1
        
    if(check == 0):
        return True
    else:
        return False