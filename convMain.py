from convNet import Network
from dataset import ImageDataset
import time
import torch
import numpy as np
import os
from PIL import Image

print("Using convNet Version", Network.__version__)

data_path = "./drive/My Drive/Images.zip"
model_path = "./drive/My Drive/models_dog/"

# data_path = "./Images.zip"
# model_path = "./models/"

seed = time.time() % 1000
seed = 421
print("SEED", seed)

dataset = ImageDataset(data_path, train=True, seed=seed, aug_rate=2, load_first_n=9999)
dataset.split(0.7)
dataset.validation_to_numpy()
dataset.shuffle_all(seed)
print(dataset.train_answers)

torch.cuda.current_device()
net = Network(dataset=dataset, learning_rate=0.00001, enable_tb=True, device="cuda:0", weight_decay=0.001,
              output_dims=dataset.n_classes)
net.set_train()
# print(net.net)
net.epoch = 0  # int(input().split()[0])
curr_model_path = model_path + "model_" + str(net.epoch) + ".pt"
print(curr_model_path)
if os.path.isfile(curr_model_path):
    net.load_model(curr_model_path)
    print("Model loaded")

net.train(savepath=model_path, train_start=-1, train_stop=-1, validation_start=-1,
          validation_stop=-1, validation_batch_size=512, batch_size=32,
          n_epochs=500, log_interval=25)
