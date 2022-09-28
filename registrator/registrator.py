import sys
sys.path.append("../reg_model")

import re
import torch
from os.path import exists
from reg_model import data_preparation as dp
from reg_model import dialog_model as dm
from torchtext.data.utils import get_tokenizer

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


def load_vocab(vocab_path):
    vocab_src, vocab_tgt = torch.load(vocab_path)
    return vocab_src, vocab_tgt

class Registrator:
    def __init__(self, vocab_path, model_path):
        EMB_SIZE = 256
        NHEAD = 8
        FFN_HID_DIM = 128
        BATCH_SIZE = 256
        NUM_ENCODER_LAYERS = 6
        NUM_DECODER_LAYERS = 6
        self.vocab_src, self.vocab_trg = load_vocab(vocab_path)
        SRC_VOCAB_SIZE = len(self.vocab_src)
        TGT_VOCAB_SIZE = len(self.vocab_trg)
        self.model = dm.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
        self.text_transform_src = dp.sequential_transforms(get_tokenizer('spacy', language='ru_core_news_sm'), 
                                               self.vocab_src, dp.tensor_transform)

        self.text_transform_trg = dp.sequential_transforms(get_tokenizer('spacy', language='ru_core_news_sm'), 
                                               self.vocab_trg, dp.tensor_transform)        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(dm.device)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Общее число обучаемых параметров: {pytorch_total_params}")
        
        
    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        src = src.to(dm.device)
        src_mask = src_mask.to(dm.device)

        memory = self.model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(dm.device)
        for i in range(max_len-1):
            memory = memory.to(dm.device)
            tgt_mask = (dm.generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(dm.device)
            out = self.model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == dp.EOS_IDX:
                break
        return ys

    def generateRegistratorAnswer(self, src_sentence):
        if(find_name(src_sentence)):
            src_sentence = "фио"
        self.model.eval()
        src = self.text_transform_src(src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(src, src_mask, max_len=128, start_symbol = dp.BOS_IDX).flatten()
        return " ".join(self.vocab_trg.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<s>", "").replace("</s>", "")    
            
# app.py        
# freechat
# registrator
# reg_model
#     dialog_model
#     data_preparation
#     nn_train