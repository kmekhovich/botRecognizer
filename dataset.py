import torch.utils.data
import numpy as np
from PIL import Image
import random
from zipfile import ZipFile
import json
import torchvision


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, path, mode="bot", seed=228, rate=10):
        how = 0
        random.seed(seed)
        if mode == "bot":
            print("[Dataset] Starting as bot...")
            self.breeds = json.loads(open('breeds_ru.json', "r", encoding="UTF-8").read())
            print("[Dataset] Breeds for bot have been loaded successfully")
            return
        self.train = np.empty(shape=[0, 3, 224, 224])
        self.traincorrect = np.empty(shape=[0])
        self.validation = np.empty(shape=[0, 3, 224, 224])
        self.validationcorrect = np.empty(shape=[0])
        print("[Dataset] Please work perfectly")
        with ZipFile(path) as zf:
            print("[Dataset] Unzipping {}".format(path))
            pictures = zf.namelist()
            self.breeds = []
            self.correct = []
            breed_id = -1
            all_images = []
            pictures.sort()
            for picture in pictures:
                if picture.endswith("/"):
                    continue
                breed_name = picture.split('/')[0][10:]
                if breed_name not in self.breeds:
                    self.breeds.append(breed_name)
                    breed_id += 1

                    print("[Dataset] Detecting new breed: {} with id {}".format(breed_name, breed_id))
                with zf.open(picture) as f:
                    img = Image.open(f)

                    ###########################
                    if random.randint(0, rate) == 0:
                        how += 1
                        img_a = img
                        img_a = self.augmentation(img_a)
                        data = np.array(img_a)
                        data = np.moveaxis(data, [0, 1], [1, 2])
                        all_images.append(data)
                        self.correct.append(breed_id)
                    ###########################

                    data = np.array(img)
                    data = np.moveaxis(data, [0, 1], [1, 2])
                    all_images.append(data)
                    self.correct.append(breed_id)
        print("[Dataset] Converting to numpy...")
        self.dataset = np.array(all_images)
        del all_images
        print("[Dataset] Successfully loaded data with shape {}".format(self.dataset.shape))
        print(how)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def correct_name(self, index):
        return self.breeds[self.correct[index]]

    def shuffle(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        positions = np.arange(len(self.dataset))
        random.shuffle(positions)
        self.dataset = self.dataset[positions]
        self.correct = np.array(self.correct)[positions].tolist()

    def augmentation(self, img):
        augment = torchvision.transforms.Compose([
            # torchvision.transforms.RandomCrop(size=10),
            torchvision.transforms.ColorJitter(brightness=.1, hue=.1, saturation=.1),
            torchvision.transforms.RandomGrayscale(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(20)
        ])
        return augment(img)