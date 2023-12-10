from matplotlib import patches
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm

from imgconvert import array_to_image, image_to_array

class CarsDataLoader(BaseDataLoader):
    """
    Cars data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True,
                 validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = CarDataSet(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         num_workers)

class CarsPartCenterDataLoader(BaseDataLoader):
    """
    Cars data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = CarPartCenter(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CarsWithBoxDataLoader(BaseDataLoader):
    """
    Cars data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = CarDataWithBox(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CarDataSet(Dataset):
    def __init__(self, img_dir, build_npy=True, transform=None, target_transform=None):
        self.num_classes = 10
        
        self.images = []
        # return all files as a list
        for file in os.listdir(img_dir):
            if file.endswith(".npy") or file.endswith(".npz"):
                self.images.append(file)
                
            
        # #### debug
        #self.images = ["0001.png","0002.png"]
            
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        # if build_npy:
        #     self.create()
    
    
    def __len__(self):
        return len(self.images)
    
    # retruns image and target (image is [C,H,W] and target is [H,W])
    # image is a float32 np.array and target uint8 array with values 0-9
    def load(self, image_name : str):
        
        
        array = np.load(self.img_dir+"/"+image_name)
        if image_name.endswith(".npz"):
            array = array["arr_0"]
        #img_path = os.path.join(self.img_dir+"/images", image_name)
        #image = cv2.imread(img_path)
        #image = np.asarray(self.resizeImage(image),np.uint8).transpose(2,0,1)
        image = np.asarray(array[:,:,:3].transpose(2,0,1)/255,dtype=np.float32)
        target = np.asarray(array[:,:,3:]/10,dtype=np.uint8).transpose(2,0,1).squeeze(0)
        return image, target

    def __getitem__(self, idx):
        image, label = self.load(self.images[idx])

        label = label
        image = torch.tensor(image,dtype=torch.float32)
        label = torch.nn.functional.one_hot(torch.tensor(label,dtype=torch.long),self.num_classes).permute(2,0,1)
        label = label.float()
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def show(self, image: np.array, target: np.array):
        target =  torch.argmax(target, dim=0)
        target = target*10
        target = np.asarray(target,dtype=np.uint8)
        target = array_to_image(target)
        target = np.asarray(target)
        image = np.asarray(image).transpose(1, 2, 0)
        target = np.asarray(target / 255, dtype=np.float32)

        # dst = cv2.addWeighted(image, 1, target , 1, 0, dtype=cv2.CV_8U)
        # cv2.imshow('image',dst)
        # cv2.waitKey(0)
        plt.imshow(image)
        plt.imshow(target, alpha=0.5)
        plt.show()
    
    def show_index(self, index: int):
        image, target = self.load(self.images[index])
        self.show(image, target)

    def resizeImage(self, image, desired_size=256):
        old_size = image.shape[:2]  # old_size is in (height, width) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format

        image = cv2.resize(image, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = 0, delta_h
        left, right = 0, delta_w

        color = [0, 0, 0]
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                  value=color)

class CarDataWithBox(CarDataSet):
    def __init__(self, img_dir, transform=None, target_transform=None):
        super().__init__(img_dir, transform, target_transform)
        self.usedParts = [1,2,3,4,5,6,7,8]

    def __getitem__(self, idx):
        image, label = self.load(self.images[idx])
        target = np.zeros((self.usedParts.__len__(),4))

        for i in range(len(self.usedParts)):

            partPixels = (label == self.usedParts[i]) # Not really necessary, can just use masses because 0 mass used as weight will work just fine.

            target[i,:] = cv2.boundingRect(partPixels[:,:].astype(np.uint8))

        target = target.flatten()
        target = torch.tensor(target,dtype=torch.float32)

        label = torch.nn.functional.one_hot(torch.tensor(label,dtype=torch.long),self.num_classes).permute(2,0,1)
        label = label.float()

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target, label



    def show(self, image, target):

        target =  torch.argmax(target, dim=0)
        target = target*10
        target = np.asarray(target,dtype=np.uint8)
        target = array_to_image(target)

        image = np.asarray(image).transpose(1,2,0)
        target = np.asarray(target/255,dtype=np.float32)

        # dst = cv2.addWeighted(image, 1, target , 1, 0, dtype=cv2.CV_8U)
        # cv2.imshow('image',dst)
        # cv2.waitKey(0)
        plt.imshow(image)
        plt.imshow(target,alpha=0.5)
        plt.show()

class CarPartCenter(CarDataSet):
    def __init__(self, img_dir, transform=None, target_transform=None):
        super().__init__(img_dir, transform, target_transform)
        self.usedParts = [1, 2, 3, 4, 5, 6, 7, 8]

    def __getitem__(self, idx):
        image, label = self.load(self.images[idx])
        
        target = np.zeros((self.usedParts.__len__(),4))
        
        for i in range(len(self.usedParts)):
            
            partPixels = (label == self.usedParts[i]) # Not really necessary, can just use masses because 0 mass used as weight will work just fine.
            
            target[i,:] = cv2.boundingRect(partPixels[:,:].astype(np.uint8))

        target = target.flatten()

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

    def boundingImages(self, boundingPoints):
        # bound box (Batch, 8*4)
        self.num_box = 8
        image = torch.zeros(boundingPoints.shape[0], self.num_box, 256, 256)
        boundingPoints = boundingPoints.view(boundingPoints.shape[0], self.num_box, 4)
        for i in range(boundingPoints.shape[0]):
            for box in range(self.num_box):
                target = boundingPoints[i,box]
                image[i,box,int(target[1]):int(target[1]+target[3]),int(target[0]):int(target[0]+target[2])] = 1
        return image.squeeze(0)

    def show(self, img, target):
        bgr_values = {
            10: (10, 150, 250),  # orange
            20: (20, 100, 20),  # dark green
            30: (10, 249, 249),  # yellow
            40: (250, 248, 10),  # cyan
            50: (150, 5, 150),  # purple
            60: (10, 250, 10),  # light green
            70: (250, 20, 20),  # blue
            80: (250, 10, 250),  # pink
            90: (100, 100, 100),
            0: (0, 0, 0),  # no color
        }
        img = np.asarray(img*255,dtype=np.uint8).transpose(1,2,0).copy()
        print(img.shape)
        rect = np.asarray(self.boundingImages(torch.tensor([target]))[0:3,:,:].numpy()*255,dtype=np.uint8).transpose(1,2,0)
        print(rect.shape)
        target = target.reshape((-1,4))
        for i in range(target.shape[0]):
            start_point = (int(target[i,0]), int(target[i,1])) 
            end_point = (int(target[i,0]+target[i,2]), int(target[i,1]+target[i,3]))
            thickness = 1
            color = bgr_values[self.usedParts[i]*10]
            print(start_point,end_point,color)
            img = cv2.rectangle(img, start_point, end_point,  color, thickness)
        img = cv2.addWeighted(img, 1, rect , 0.5, 0, dtype=cv2.CV_8U)
        cv2.imshow('image',img)
        cv2.waitKey(0)

    def show_index(self, index: int):
        #BGR hood, front door, rear door, frame, rear quater panel, trunk lid, fender, bumper
        img, target = self.__getitem__(index)
        self.show(img, target)


