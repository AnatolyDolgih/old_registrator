import torch
import my_model as tr_model
import data_preparation as dp
from torchtext.data.utils import get_tokenizer
from timeit import default_timer as timer

SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = dp.get_vocab_size("./vocab.pt")
EMB_SIZE = 256
NHEAD = 8
FFN_HID_DIM = 128
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

vocab_src, vocab_trg = torch.load("vocab.pt")

text_transform_src = dp.sequential_transforms(get_tokenizer('spacy', language='ru_core_news_sm'), 
                                               vocab_src, dp.tensor_transform)

text_transform_trg = dp.sequential_transforms(get_tokenizer('spacy', language='ru_core_news_sm'), 
                                               vocab_trg, dp.tensor_transform)

transformer = tr_model.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

transformer.load_state_dict(torch.load("./Models/model_params_10_epoch.pth"))
transformer.eval()
transformer.to(tr_model.device)

# visitor_phrazes= ['Извините, я тороплюсь.', 'Добрый вечер.', 'Да, я хочу у вас остановиться.', 
#                   'Бронь на меня должна быть, но номер я не знаю. Бронировал(а) не я.', 'Абрамов Тимур Михайлович',
#                   'Подтвердить не могу, но мне сказали, что номер будет забронирован.', 'Да, пожалуйста.', 'Одна ночь.',
#                   'Мне нужен номер на одного человека.', 'Спасибо, я постою.', 'Нет, я никуда не спешу.', 'Не имеет значения.',
#                   'King (очень большой, 193 X 203 см)', 'Спасибо, хорошее.', 'Да, я бы рассмотрел(а) другие варианты.',
#                   'День был прекрасный.', 'Да, очень.', 'Хорошо, давайте продолжим оформление.', 'Узнаю в другой раз.']

# time = []
# for phraze in visitor_phrazes:
#     start_time = timer()
#     print(f"Посетитель >> {phraze}")
#     print(f"Регистратор >> {tr_model.translate(transformer, phraze, vocab_trg, text_transform_src)}")
#     end_time = timer()
#     time.append(end_time - start_time)

# print(f"среднее время для обработки запроса = {sum(time)/len(time):.3f}s")

import socket

server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()
print(host)
server_sock.bind((host, 1080))
server_sock.listen(100)
print("Server was run")
client_sock, addr = server_sock.accept()
print(f"connection from: {addr}")
count = 0
while True:
    data = client_sock.recv(10000).decode()
    if not data:
        break
    count += 1
    print(f"{count} >> recieved message: {str(data)}")        

    answer = tr_model.translate(transformer, str(data), vocab_trg, text_transform_src)
    string = answer.ljust(500, " ")
    print(string)
    print(len(string))
    print(len(string))
    client_sock.send(string.encode('utf-8'))
    print(f"{count} >> send message: Электричество")

print(count)
client_sock.close()
server_sock.close()

#  Чтобы одинаковые предложения не повторялись
#  Не отвечать на повторяющиеся сообщения

