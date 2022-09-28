import time
import sys
sys.path.append("../")

from classificator import classificator as cls
from freechat import freechat as fch 
from registrator import registrator as reg

# lines = []

# all_time = []
# max_time = 0.0
# min_time = 0.0

# for _ in range(0, 500):
#     start = time.time()
#     cls.classify("здесь неплохо")
#     end = time.time()
#     all_time.append(end - start)

# for _ in range(0, 500):
#     start = time.time()
#     cls.classify("тебе нравится фантастика?")
#     end = time.time()
#     all_time.append(end - start)

# max_time = max(all_time)
# min_time = min(all_time)    
# avg_time = sum(all_time) / 1000

# lines.append((f"avg time: {avg_time:.3g} max_time: {max_time:.3g} min_time: {min_time:.5g}"))

####################################################################################    

# chatbot = fch.FreeChatBot()

# for _ in range(0, 500):
#     start = time.time()
#     answer = chatbot.generateFreeAnswer("тебе нравится фантастика?")
#     end = time.time()
#     all_time.append(end - start)
#     print(answer)

# max_time = max(all_time)
# min_time = min(all_time)    
# avg_time = sum(all_time) / 500

# lines.append((f"avg time: {avg_time:.3g} max_time: {max_time:.3g} min_time: {min_time:.5g}"))
# print(lines)

###################################################################################

# registrator = reg.Registrator("../reg_model/vocab.pt", "../reg_model/saved_model/model_params_20_epoch.pth")

# for _ in range(0, 500):
#     start = time.time()
#     answer = registrator.generateRegistratorAnswer("здесь неплохо")
#     end = time.time()
#     all_time.append(end - start)
#     print(answer)

# max_time = max(all_time)
# min_time = min(all_time)    
# avg_time = sum(all_time) / 500

# lines.append((f"avg time: {avg_time:.3g} max_time: {max_time:.3g} min_time: {min_time:.5g}"))
# print(lines)

###################################################################################

lines = []

all_time = []
max_time = 0.0
min_time = 0.0

for i in range(0, 250):
    lines.append("здесь неплохо")
    lines.append("ты любишь фантастику")


chatbot = fch.FreeChatBot("../freechat/RuDialoGPT/model")
registrator = reg.Registrator("../reg_model/vocab.pt", "../reg_model/saved_model/model_params_20_epoch.pth")

count = 0
for replic in lines:
    start = time.time()
    if (replic == "exit"):
        break
    if(cls.classify(replic)):
        registrator.generateRegistratorAnswer(replic)
    else:
        chatbot.generateFreeAnswer(replic)
    end = time.time()
    all_time.append(end - start)
    print(count)
    count += 1    
max_time = max(all_time)
min_time = min(all_time)    
avg_time = sum(all_time) / 1000

print(((f"avg time: {avg_time:.3g} max_time: {max_time:.3g} min_time: {min_time:.5g}")))
