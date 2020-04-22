import os
import time
import torch
import numpy as np
import torchvision.transforms as transforms

from .albumentation import CustomAlbumentation
from .dataset import CustomDataset


imagenet_root = 'tiny-imagenet-200'

mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
cuda = False

train_transform = CustomAlbumentation(mean, std)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
)


def check_cuda():
    SEED = 1

    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    
    # For reproducibility
    torch.manual_seed(SEED)
    
    if cuda:
        torch.cuda.manual_seed(SEED)


def get_id_dictionary():
    id_dict = {}
    for i, line in enumerate(open(imagenet_root + '/wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict
    

def get_train_test_data(id_dict, test_split):
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    t = time.time()
    total_val_images = 10000
    images_for_class = 500
    
    train_images_count = images_for_class - (images_for_class * test_split)
    test_images_count = total_val_images - (total_val_images * test_split)
    
    for key, value in id_dict.items():
        all_data, all_labels = [], []
        for i in range(images_for_class):
            all_data.append(imagenet_root + f'/train/{key}/images/{key}_{str(i)}.JPEG')
            all_labels.append(id_dict[key])

        for x in range(0, images_for_class):
            if x < train_images_count:
                train_data.append(all_data[x])
                train_labels.append(all_labels[x])
            else:
                test_data.append(all_data[x])
                test_labels.append(all_labels[x])

    val_count = 0
    for line in open(imagenet_root + '/val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        if val_count < test_images_count:
            train_data.append(imagenet_root + f'/val/images/{img_name}')
            train_labels.append(id_dict[class_id])
        else:
            test_data.append(imagenet_root + f'/val/images/{img_name}')
            test_labels.append(id_dict[class_id])

        val_count += 1

    print(f'Finished loading data in {time.time() - t} seconds.')
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)
    

def get_data_loader(train_data, train_labels, test_data, test_labels):
    trainset = CustomDataset(train_data, train_labels, train_transform)
    testset = CustomDataset(test_data, test_labels, test_transform)
    
    dataloader_args = dict(shuffle=True, batch_size=256, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
        
    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
    
    return trainloader, testloader
