import torch
import torch.nn as nn
from tensorboardcolab import TensorBoardColab
import numpy as np
import ftplib
import torchvision
import time

print("Using PyTorch Version", torch.__version__)
print("Using TorchVision Version", torchvision.__version__)
if torch.cuda.is_available():
    print("Cuda is available")


class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.reshape(inputs.shape[0], -1)


class Network:

    __version__ = "-1.1.0"

    def __init__(self, dataset=None, learning_rate=0.01, enable_tb=False, device="cpu"):
        self.device = device
        print("Using", self.device)
        self.learning_rate = learning_rate
        self.training_loss = []
        self.training_acc = []
        self.last_validation_loss = 0.0
        self.last_validation_acc = 0.0
        self.last_tb_update = time.time()
        self.epoch = 0
        self.enable_tb = enable_tb
        self.dataset = dataset

        if self.enable_tb:
            self.tb = TensorBoardColab()

        self.net = torchvision.models.resnet50(pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.fc = nn.Sequential(
                                    nn.Linear(2048, 120),
                                    nn.Softmax(1))

        if self.device != "cpu":
            self.net.to(self.device)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            self.net.cuda()
            self.gpu = 1
        else:
            self.gpu = 0

        self.target_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.learning_rate)

    def set_train(self):
        self.net.train()

    def set_eval(self):
        self.net.eval()

    def start_ftp(self, host, username, password):
        self.ftp = ftplib.FTP(host, username, password)

    def save_model_ftp(self, path="/models/model.pt", filename="model.pt"):
        with open(filename, 'rb') as fobj:
            self.ftp.storbinary('STOR ' + path, fobj, 1024)

    def close_ftp(self):
        self.ftp.close()

    def send_to_device(self, arr):
        if type(arr) == torch.Tensor and str(arr.device) == self.device:
            return arr
        arr_ = torch.FloatTensor(arr)
        if self.gpu:
            arr_ = arr_.to(device=self.device)
        return arr_

    def predict(self, inputs):
        return self.net(self.send_to_device(inputs))

    def predict_num(self, inputs):
        # TODO: not working
        p = self.predict(inputs)
        print(p)
        return p.item()

    def calc_loss(self, predicts, correct_answers):
        return self.target_func(predicts, correct_answers.long())

    @staticmethod
    def calc_accuracy(predicts, correct_answers):
        return float((predicts.argmax(1) == correct_answers.long()).sum(dtype=torch.float) / predicts.shape[0]) * 100

    def update_net(self, predicts, correct_answers):
        self.optimizer.zero_grad()
        loss = self.calc_loss(predicts, correct_answers)
        self.training_loss.append(loss.item())
        loss.backward()
        self.optimizer.step()

    def update_tb(self, last_training_loss, last_training_acc):
        self.tb.save_value("Loss", "Validation Loss", self.epoch, self.last_validation_loss)
        self.tb.flush_line("Validation Loss")
        self.tb.close()
        # print("Validation loss saved", self.epoch, self.last_validation_loss)

        self.tb.save_value("Accuracy", "Validation Accuracy, %", self.epoch, self.last_validation_acc)
        self.tb.flush_line("Validation Accuracy, %")
        self.tb.close()
        # print("validation acc saved", self.epoch, self.last_validation_acc)

        self.tb.save_value("Loss", "Train Loss", self.epoch, last_training_loss)
        self.tb.flush_line("Train Loss")
        self.tb.close()
        # print("Training loss saved", self.epoch, self.last_training_loss)

        self.tb.save_value("Accuracy", "Train Accuracy, %", self.epoch, last_training_acc)
        self.tb.flush_line("Train Accuracy, %")
        self.tb.close()
        # print("Training acc saved", self.epoch, self.last_training_acc)

    def train_on_single_batch(self, inputs, correct_answers):
        inputs = self.send_to_device(inputs)
        correct_answers = self.send_to_device(correct_answers)
        predicts = self.predict(inputs)
        self.update_net(predicts, correct_answers)
        self.training_acc.append(self.calc_accuracy(predicts, correct_answers))


    def train(self, train_start, train_stop, validation_start, validation_stop, savepath,
              validation_batch_size=500, batch_size=4000, n_epochs=1000, log_interval=10):
        for _ in range(n_epochs):
            self.training_loss = []
            self.training_acc = []
            now = train_start
            last_logged = 0
            while now < train_stop - batch_size:
                batch_start = now
                batch_end = now + batch_size
                now += batch_size
                self.train_on_single_batch(self.dataset[batch_start:batch_end], self.dataset.correct[batch_start:batch_end])
                if last_logged >= log_interval:
                    print('%f%% [%d / %d]. Train accuracy: %f%%. Train loss: %f. Epoch %d' % ((100.0 * batch_end) / train_stop,
                                    batch_end, train_stop, self.training_acc[-1], self.training_loss[-1], self.epoch))
                    last_logged = 0
                last_logged += 1
            self.epoch += 1
            filename = savepath + "model_" + str(self.epoch) + ".pt"
            self.save_model(filename)
            print("Model saved:", filename)
            self.last_validation_loss, self.last_validation_acc = \
                self.test(validation_start, validation_stop, validation_batch_size)
            print('Validation accuracy: %f%%. Validation loss: %f' % (self.last_validation_acc,
                                                                      self.last_validation_loss))
            training_loss = float(np.mean(self.training_loss))
            training_acc = float(np.mean(self.training_acc))
            print('Training accuracy: %f%%. Training loss: %f' % (training_acc, training_loss))
            self.update_tb(training_loss, training_acc)

    def save_model(self, path="model.pt"):
        torch.save(self.net.state_dict(), path)

    def load_model(self, path="model.pt"):
        self.net.load_state_dict(torch.load(path))

    def test(self, validation_start, validation_stop, batch_size=500):
        loss = []
        acc = []
        now = validation_start
        while now + batch_size < validation_stop:
            batch_start = now
            batch_end = now + batch_size
            now += batch_size

            inputs = self.send_to_device(self.dataset[batch_start:batch_end])
            correct_answers = self.send_to_device(self.dataset.correct[batch_start:batch_end])
            predicts = self.predict(inputs)

            loss.append(self.calc_loss(predicts, correct_answers).item())
            acc.append(self.calc_accuracy(predicts, correct_answers))
        return float(np.mean(loss)), float(np.mean(acc))

    @staticmethod
    def find_max_indexes(arr, n_top):
        return (-arr).argsort()[:n_top]

    def answer(self, inputs, n_top):
        inputs = self.send_to_device(inputs)
        n_inputs = inputs.shape[0]
        predicts = self.predict(inputs)
        result_s = []
        result_i = []
        for i in range(n_inputs):
            indexes = self.find_max_indexes(predicts[i], n_top)
            ans_s = []
            ans_i = []
            for j in indexes:
                j = j.item()
                ans_s.append(self.dataset.breeds[j])
                ans_i.append(predicts[i][j].item() * 100.0)
            result_s.append(ans_s)
            result_i.append(ans_i)
        del inputs
        del predicts
        return result_s, result_i
