import os
import cv2
import glob
import numpy as np
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
label_name = ['airplane','automobile','bird', 'cat', 'deer','dog','frog','horse','ship','truck']
save_path = './data/cifar-10-batches-py/test'
train_list = glob.glob('./data/cifar-10-batches-py/test_batch')
for l in train_list:
    l_dict = unpickle(l)
    for im_idx, im_data in enumerate(l_dict[b'data']):
        im_label = l_dict[b'labels'][im_idx]
        im_name = l_dict[b'filenames'][im_idx]

        im_label_name = label_name[im_label]
        im_data = np.reshape(im_data,[3,32,32])
        im_data = np.transpose(im_data, (1,2,0))

        # cv2.imshow("im_data", cv2.resize(im_data, (200,200)))
        # cv2.waitKey(0)

        if not os.path.exists("{}/{}".format(save_path,im_label_name)):
            os.mkdir("{}/{}".format(save_path,im_label_name))
        cv2.imwrite("{}/{}/{}".format(save_path,im_label_name,im_name.decode("utf-8")), im_data)

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

label_dict = {}
for idx, name in enumerate(label_name):
    label_dict[name] = idx


def default_loader(path):
    return Image.open(path).convert("RGB")

class MyDataset(Dataset):
    def __init__(self, im_list, transform = None, loader = default_loader):
        super(Dataset, self).__init__()

        imgs = []
        for im_item in im_list:
            im_item = im_item.replace('\\', '/')
            
            im_label_name = im_item.split('/')[-2]
            imgs.append([im_item, label_dict[im_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader


    
    def __getitem__(self, index):

        im_path, im_label = self.imgs[index]
        im_data = self.loader(im_path)
        if self.transform is not None:
            im_data = self.transform(im_data)
        return im_data, im_label

    def __len__(self):
        return len(self.imgs)


