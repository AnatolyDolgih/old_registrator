import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class FreeChatBot:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.flag = True
        self.count = 0

    def get_length_param(self, text):
        tokens_count = len(self.tokenizer.encode(text))
        if tokens_count <= 15:
            len_param = '1'
        elif tokens_count <= 50:
            len_param = '2'
        elif tokens_count <= 256:
            len_param = '3'
        else:
            len_param = '-'
        return len_param

    def generateFreeAnswer(self, input_replic):
        new_user_input_ids = self.tokenizer.encode(f"|0|{self.get_length_param(input_replic)}|" + 
                                                   input_replic + self.tokenizer.eos_token +  "|1|1|", return_tensors="pt")
                
        bot_input_ids = new_user_input_ids
        
        chat_history_ids = self.model.generate(
            bot_input_ids,
            num_return_sequences=1,
            max_length=512,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature = 0.6,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        result = self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        return result

# test
if(__name__ == '__main__'):
    chatbot = FreeChatBot()
    print("i'm ready to chat")
    print(chatbot.generateFreeAnswer("Привет")) 
    
