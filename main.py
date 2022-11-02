from model import Bottleneck,ResNet
from myDataset import MyDataset
import os
import cv2
import glob
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from torch import nn
from PIL import Image
from matplotlib import pyplot as plt

if __name__=="__main__":
    batch_size = 128
    epochs = 200
    lr = 0.01
    trainc = []
    testc = []
    x = []
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((28, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.RandomGrayscale(0.1),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.ToTensor()
    ])
    train_list = glob.glob(r"./data/cifar-10-batches-py/train/*/*.png")
    test_list = glob.glob(r"./data/cifar-10-batches-py/test/*/*.png")


    train_dataset = MyDataset(train_list, transform=train_transform)
    test_dataset = MyDataset(test_list, transform=transforms.ToTensor())


    train_dataLoader = DataLoader(train_dataset, batch_size, True, num_workers=4)
    test_dataLoader = DataLoader(test_dataset, batch_size, False, num_workers=4)

    model = ResNet().to(device)

    loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr = lr )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    for epoch in range(epochs):
        print("epoch is ", epoch)
        x.append(epoch)
        sum_correct = 0.
        for i, data in enumerate(train_dataLoader):
            print("batch is" , i)
            model.train()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_func(outputs, labels.data)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            _, pred = torch.max(outputs.data, dim=1)

            correct = pred.eq(labels.data).cpu().sum()
            sum_correct += correct
            print("batch",i, "loss is:", loss.item(),
            "correct is:", 100.0 * float(correct)/batch_size)
        train_correct = sum_correct / len(train_dataLoader)/batch_size
        trainc.append(train_correct*100.)
        print("epoch ", epoch, "train correct is ", 100. * train_correct)
        scheduler.step()
        print("lr is ", optimizer.state_dict()["param_groups"][0]["lr"])

        sum_correct = 0.
        for i, data in enumerate(test_dataLoader):
            model.eval()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, pred = torch.max(outputs.data, dim=1)

            correct = pred.eq(labels.data).cpu().sum()
            sum_correct += correct
            print("batch",i,
            "test correct is:", 100.0 * correct/batch_size)

        test_correct = sum_correct / len(test_dataLoader)/batch_size
        testc.append(test_correct*100.)
        print("epoch ", epoch, "test correct is ", 100. * test_correct)
        if epoch % 50 == 0 :
            plt.plot(x, trainc, label = 'train accuracy', color = 'blue')
            plt.plot(x, testc, label = 'test accuracy', color = 'red', linestyle = '--')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('accuracy(%)')
            plt.show(block = False)
            plt.pause(10)
            
            plt.close()
        
