import torch
import telebot
import numpy as np
from convNet import Network
from dataset import ImageDataset
from PIL import Image
import requests
import system
from urllib.parse import quote


TOKEN = '773221186:AAF7xpKcOCmsbq6DnKG9ce5mxg21E4VcNIk'
bot = telebot.TeleBot(TOKEN)
# torch.cuda.current_device()
net = Network(device="cuda:0")
net.set_eval()
model_path = "./model_21.pt"
net.load_model(model_path)
data_path = "Images.zip"
database = ImageDataset(zip_path=data_path, json_path="./breeds_ru.json")
net.dataset = database
# system.manage_downloading(net)


@bot.message_handler(commands=['start'])
def newmes(message):
    bot.reply_to(message, "Для распознавания породы собаки отправь фото")


@bot.message_handler(content_types=['photo'])
def recog(message):
    url = "https://api.telegram.org/file/bot{}/{}".format(TOKEN, bot.get_file(message.photo[-1].file_id).file_path)
    temp_path = 'temp.jpg'
    with open(temp_path, 'wb') as handle:
        response = requests.get(url, stream=True)
        for block in response.iter_content(1024):
            if not block:
                break
            handle.write(block)
    img = database.normalize([Image.open(temp_path)])
    data = torch.stack(img)
    str = ""
    n_top = 5
    ans = net.answer(data, n_top)
    for i in range(n_top):
        str = "{}\n{}. {}: {}%".format(str, i + 1, ans[0][0][i], round(ans[1][0][i], 2))
    bot.reply_to(message, str)
    path = "Preview/{}.jpg".format(ans[0][0][0])
    with open(path, 'rb') as im_f:
        bot.send_photo(message.chat.id, im_f, caption=ans[0][0][0])


bot.polling()
