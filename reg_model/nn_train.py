import torch
import torch.nn as nn
import dialog_model as dm
import data_preparation as dp
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, save, output_file
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from progress.bar import ChargingBar 
from timeit import default_timer as timer
import openpyxl

from torch.utils.data import DataLoader

from GPUtil import showUtilization as gpu_usage
import pandas as pd

file_name = "loss_visualization.html"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Подготовка данных
train_data, valid_data = dp.readAndSplitData("./Data/RegistrationDialogs.txt", 0.75)

vocab_src, vocab_trg = dp.load_vocab(dp.spacy_ru, train_data, valid_data)

src_vocab_size = len(vocab_src.vocab)
trg_vocab_size = len(vocab_trg.vocab)    

train_iter = dp.MyDataset(train_data)
valid_iter = dp.MyDataset(valid_data)

text_transform_src = dp.sequential_transforms(get_tokenizer('spacy', language='ru_core_news_sm'), 
                                               vocab_src, dp.tensor_transform)

text_transform_trg = dp.sequential_transforms(get_tokenizer('spacy', language='ru_core_news_sm'), 
                                               vocab_trg, dp.tensor_transform)

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform_src(src_sample.rstrip("\n")))
        tgt_batch.append(text_transform_trg(tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=dp.PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=dp.PAD_IDX)
    return src_batch, tgt_batch

print(len(train_data))
print(len(valid_data))


SRC_VOCAB_SIZE = len(vocab_src)
TGT_VOCAB_SIZE = len(vocab_trg)
EMB_SIZE = 256
NHEAD = 8
FFN_HID_DIM = 128
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

NUM_EPOCHS = 1

transformer = dm.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=dp.PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-8)

def save_nn_in_epoch(model, epoch):
    name = "saved_model/model_" + str(epoch) + "_epoch.pth"
    name_param = "saved_model/model_params_" + str(epoch) + "_epoch.pth"
    torch.save(model, name)
    torch.save(model.state_dict(), name_param)


def train_epoch(model, optimizer):
    model.train()
    losses = 0.0
    
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    bar = ChargingBar("Тренировка ...", max = len(train_dataloader))
    
    for src, tgt in train_dataloader:
        bar.next()
        src = src.type(torch.LongTensor)
        tgt = tgt.type(torch.LongTensor)
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = dm.create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        del src, tgt
    bar.finish()
    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0.0

    valid_dataloader = DataLoader(valid_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    for src, tgt in valid_dataloader:
        src = src.type(torch.LongTensor)
        tgt = tgt.type(torch.LongTensor)
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = dm.create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
        del src, tgt
    return losses / len(valid_dataloader)

def train():
    epochs = []
    train_l = []
    valid_l = []
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        
        valid_loss = evaluate(transformer)
        save_nn_in_epoch(transformer, epoch)
        
        epochs.append(epoch)
        train_l.append(train_loss)
        valid_l.append(valid_loss)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, \
               Valid loss: {valid_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    df = pd.DataFrame({
        "Epochs": epochs,
        "Train_loss": train_l,
        "Valid_loss": valid_l,
    })
    source = ColumnDataSource(df)
    plot = figure(title = "Зависимость ошибки на тренировочной и валидационной выборках от эпохи",
                  max_width=750, height=500,
                  x_axis_label = "Эпоха",
                  y_axis_label = "Ошибка")
    plot.line(x= 'Epochs', y='Train_loss',
              color='red', alpha=0.8, legend="Train loss", line_width=2,
              source=source)
    plot.line(x= 'Epochs', y='Valid_loss',
              color='blue', alpha=0.8, legend='Valid loss', line_width=2,
              source=source)
    output_file(filename = file_name)
    df.to_excel("train_stat_excel.xlsx", index=False)
    df.to_csv("train_stat_csv.csv", index=False, sep=";")
    save(plot)
    print("Нейронная сеть обучилась")



torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)
torch.manual_seed(0)

train()