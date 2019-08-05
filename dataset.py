import torch.utils.data
import numpy as np
from PIL import Image
import os
from zipfile import ZipFile
import json
import torchvision
import time
import math
import torchvision.transforms as TT


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, zip_path, train=False, pic_size=224, json_path="", load_first_n=99999, debug=0, aug_rate=3,
                 seed=math.floor(time.time() % 1000)):
        np.random.seed(seed)
        if os.path.isfile(json_path):
            self.class_names = json.loads(open(json_path, "r", encoding="UTF-8").read())
            print("[Dataset] Imported json with {} class names".format(len(self.class_names)))
        else:
            print("[Dataset] Can't load json with class names")
        self.seed = seed
        self.train = [] # np.empty(shape=[0, 3, size, size])
        self.train_answers = []
        self.validation = []
        self.validation_answers = []
        self.size = 0
        self.aug_rate = aug_rate
        self.normalization = TT.Compose([
            TT.Resize((pic_size, pic_size)),
            TT.ToTensor(),
            TT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # TODO: torchvision random seed
        self.augment = TT.Compose([
            # torchvision.transforms.RandomCrop(size=10),
            TT.ColorJitter(brightness=.1, hue=.1, saturation=.1),
            TT.RandomGrayscale(),
            TT.RandomHorizontalFlip(),
            TT.RandomRotation(20),
        ])
        if not train:
            return
        # TODO: loading in function
        with ZipFile(zip_path) as zf:
            if debug: print("[Dataset] Unzipping {}".format(zip_path))
            pictures_list = zf.namelist()
            pictures_list.sort()
            self.classes = []
            buffered_pics = []
            for picture_name in pictures_list:
                if picture_name.endswith("/"):
                    if len(buffered_pics) > 0:
                        self.classes.append(buffered_pics.copy())
                    buffered_pics.clear()
                    class_name = picture_name.split('-')[1][:-1]
                    if debug: print(
                        "[Dataset] Detecting new class: \"{}\" with id {}".format(class_name, len(self.classes)))
                    if len(self.classes) >= load_first_n:
                        break
                    continue
                with zf.open(picture_name) as f:
                    buffered_pics.append(Image.open(f).copy())
                    self.size += 1
            if len(buffered_pics) > 0:
                self.classes.append(buffered_pics.copy())
        print("[Dataset] Successfully loaded data with {} classes and size {}".format(len(self.classes), self.size))

    def __getitem__(self, index):
        imgs = self.train[index]
        if str(type(index)) == "<class 'int'>":
            imgs = [imgs]
        return torch.stack(self.normalize(self.augmentation(imgs)))

    def __len__(self):
        return self.size

    def get_name(self, index):
        return self.class_names[index]

    def split(self, rate=0.7):
        for class_num, class_ in enumerate(self.classes):
            train_n = math.floor(rate * len(class_))

            for j in range(train_n):
                self.train.append(class_[j])
                self.train_answers.append(class_num)

            for j in range(len(class_) - train_n):
                self.validation.append(class_[j])
                self.validation_answers.append(class_num)
        self.classes.clear()

    def validation_to_numpy(self):
        print("[Dataset] Converting validation array to numpy...")
        self.validation = torch.stack(self.normalize(self.validation))
        print("[Dataset] Ready")

    @staticmethod
    def shuffle(arr, seed=math.floor(time.time() % 1000)):
        nums = np.arange(len(arr))
        np.random.seed(seed)
        np.random.shuffle(nums)
        for i in range(len(arr)):
            if nums[i] != i:
                arr[i], arr[nums[i]] = arr[nums[i]], arr[i]
                temp = nums[nums[i]]
                nums[nums[i]] = nums[i]
                nums[i] = temp

    def shuffle_train(self, seed=math.floor(time.time() % 1000)):
        self.shuffle(self.train, seed)
        self.shuffle(self.train_answers, seed)

    def shuffle_validation(self, seed=math.floor(time.time() % 1000)):
        self.shuffle(self.validation, seed)
        self.shuffle(self.validation_answers, seed)

    def shuffle_all(self, seed=math.floor(time.time() % 1000)):
        self.shuffle_train(seed)
        self.shuffle_validation(seed)

    @staticmethod
    def show(img):
        img.show()

    def normalize(self, imgs):
        # TODO: Maybe remove loop
        for i in range(len(imgs)):
            imgs[i] = self.normalization(imgs[i])
        return imgs

    def augmentation(self, imgs):
        # TODO: Maybe remove loop
        for i in range(len(imgs)):
            if np.random.randint(0, self.aug_rate) == 0:
                imgs[i].show()
                imgs[i] = self.augment(imgs[i])
                imgs[i].show()
        return imgs
