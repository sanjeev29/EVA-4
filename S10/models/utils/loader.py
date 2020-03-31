import torch
import torchvision
import torchvision.transforms as transforms

from .albumentation import CustomAlbumentation


mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

# Train set with albumentation
trainset_with_alb = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=CustomAlbumentation(mean, std))

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

dataloader_args_for_plot = dict(shuffle=True, batch_size=4, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# train dataloader
trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
trainloader_for_plot = torch.utils.data.DataLoader(trainset, **dataloader_args_for_plot)

# train dataloader for albumentation
trainloader_alb = torch.utils.data.DataLoader(trainset_with_alb, **dataloader_args)
trainloader_for_plot_alb = torch.utils.data.DataLoader(trainset_with_alb, **dataloader_args_for_plot)

# test dataloader
testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
testloader_for_plot = torch.utils.data.DataLoader(testset, **dataloader_args_for_plot)
