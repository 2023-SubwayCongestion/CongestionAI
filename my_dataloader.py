from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from transforms import Transforms
from torchvision.transforms import functional
from PIL import Image


class CrowdDataset(Dataset):
    '''
    crowdDataset
    '''
    def __init__(self,img_root,gt_dmap_root, gt_downsample=1, dataset_name = "SHA", is_train = False):
        '''
        img_root: the root path of img.
        gt_dmap_root: the root path of ground-truth density-map.
        gt_downsample: default is 0, denote that the output of deep-model is the same size as input image.
        '''
        self.img_root=img_root
        self.gt_dmap_root=gt_dmap_root
        self.gt_downsample=gt_downsample

        self.img_names=[filename for filename in os.listdir(img_root) \
                           if os.path.isfile(os.path.join(img_root,filename))]
        self.n_samples=len(self.img_names)

        self.dataset_name = dataset_name
        self.is_train = is_train

    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'
        image_name=self.img_names[index]

        image = Image.open(os.path.join(self.img_root,image_name)).convert('RGB')
        # image=plt.imread(os.path.join(self.img_root,image_name))
        trans = Transforms((0.8, 1.2), (400, 400), self.gt_downsample, (0.5, 1.5), self.dataset_name)
        # if len(image.size)==2: # expand grayscale image to three channel.
        #     image=image[:,:,np.newaxis]
        #     image=np.concatenate((image,image,image),2)

        gt_dmap=np.load(os.path.join(self.gt_dmap_root,image_name.replace('.jpg','.npy')))
        

        density = np.array(gt_dmap, dtype=np.float32)
        if self.is_train:
            image, density = trans(image, density)
            return image, density
        else:
        #     height, width = image.size[1], image.size[0]
        #     height = round(height / 16) * 16
        #     width = round(width / 16) * 16
        #     image = image.resize((width, height), Image.BILINEAR)

        #     image = functional.to_tensor(image)
        #     # img_tensor=torch.tensor(img,dtype=torch.float)
        #     image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     gt_dmap_tensor=torch.tensor(gt_dmap,dtype=torch.float)
            
        #     return image, gt_dmap_tensor


            # if self.gt_downsample>1: # to downsample image and density-map to match deep-model.
            ds_rows=int(image.size[0]//self.gt_downsample)
            ds_cols=int(image.size[1]//self.gt_downsample)
            image = image.resize((ds_rows*self.gt_downsample, ds_cols*self.gt_downsample), Image.BILINEAR)
            # image = cv2.resize(image,(ds_cols*self.gt_downsample,ds_rows*self.gt_downsample))
            # image=image.transpose((2,0,1)) # convert to order (channel,rows,cols)

            gt_dmap=cv2.resize(gt_dmap,(ds_rows, ds_cols))
            gt_dmap=gt_dmap[np.newaxis,:,:]*self.gt_downsample*self.gt_downsample

            img_tensor= functional.to_tensor(image)
            # img_tensor = img_tensor.permute(0, 2, 1)
            gt_dmap_tensor=torch.tensor(gt_dmap,dtype=torch.float)
            # gt_dmap_tensor = gt_dmap_tensor.permute(0, 2, 1)

        return img_tensor,gt_dmap_tensor


# test code
if __name__=="__main__":
    train_img_root="/home/add/dataset/ShanghaiTech/part_A_final/train_data/images"
    train_gt_dmap_root="/home/add/dataset/ShanghaiTech/part_A_final/train_data/ground-truth"

    test_img_root="/home/add/dataset/ShanghaiTech/part_A_final/test_data/images"
    test_gt_dmap_root="/home/add/dataset/ShanghaiTech/part_A_final/test_data/ground-truth"
    
    train_dataset=CrowdDataset(train_img_root,train_gt_dmap_root, gt_downsample = 4, dataset_name = 'SHA', is_train = True)
    for i,(img,gt_dmap) in enumerate(train_dataset):
        # plt.imshow(img.permute(1, 2, 0))
        # plt.figure()
        # plt.imshow(gt_dmap[0,:,:], cmap='gray')
        # plt.figure()
        # if i>5:
        #     break
        print(img.shape,gt_dmap.shape)

    

    test_dataset=CrowdDataset(test_img_root,test_gt_dmap_root, gt_downsample = 4, dataset_name = 'SHA', is_train = False)
    for i,(img,gt_dmap) in enumerate(test_dataset):
        # plt.imshow(img.permute(1, 2, 0))
        # plt.figure()
        # plt.imshow(gt_dmap[0,:,:], cmap='gray')
        # plt.figure()
        # if i>5:
        #     break
        print(img.shape,gt_dmap.shape)