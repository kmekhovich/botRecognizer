import telebot
import numpy as np
from convNet import Network
from dataset import ImageDataset
from PIL import Image
import requests
import system

print(system.getmemory(3))

TOKEN = '773221186:AAF7xpKcOCmsbq6DnKG9ce5mxg21E4VcNIk'
bot = telebot.TeleBot(TOKEN)
net = Network(device="cuda:0")
net.set_eval()
model_path = "model_156.pt"
net.load_model(model_path)
data_path = "Images.zip"
database = ImageDataset(data_path)
net.dataset = database
dwnld = 0
if dwnld == 1:
    system.manage_downloading(net)

print(system.getmemory(3))


@bot.message_handler(commands=['start'])
def newmes(message):
    bot.reply_to(message, "Для распознавания породы собаки отправь фото")


@bot.message_handler(content_types=['photo'])
def recog(message):
    print(system.getmemory(3))
    url = "https://api.telegram.org/file/bot{}/{}".format(TOKEN, bot.get_file(message.photo[-1].file_id).file_path)
    temp_path = 'temp.jpg'
    with open(temp_path, 'wb') as handle:
        response = requests.get(url, stream=True)
        for block in response.iter_content(1024):
            if not block:
                break
            handle.write(block)
    img = Image.open(temp_path)
    img = img.resize((224, 224), Image.ANTIALIAS)
    data = np.array(img)
    data = np.moveaxis(data, [0, 1], [1, 2])
    str = ""
    n_top = 5

    ans = net.answer([data], n_top)
    for i in range(n_top):
        str = "{}\n{}. {}: {}%".format(str, i + 1, ans[0][0][i], round(ans[1][0][i], 2))
    bot.reply_to(message, str)
    path = "Preview/{}.jpg".format(ans[0][0][0])
    with open(path, 'rb') as im_f:
        bot.send_photo(message.chat.id, im_f, caption=ans[0][0][0])
    del img
    del data

    print(system.getmemory(3))


bot.polling()
