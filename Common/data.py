from torch.utils.data import DataLoader #PyTorch's btaching + shuffling + loading engine
from torchvision import datasets, transforms 
from torch.utils.data import Subset



#datasets provdies ready-made datasets like CIFAR-10
#trasnforms defines how raw images are processed before the model sees them

def get_dataloaders(batch_size = 128, num_workers = 0, aug_strength="weak"):

#batch_size - how many images per batch
#num_workers - how many CPU processes load data in parallel

    if aug_strength == "weak":
        train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding = 4), #data augmentation
        transforms.RandomHorizontalFlip(), #randomly flips images 
        transforms.ToTensor(),
        transforms.Normalize( #normalizes data, Xnormalized = (x - mean)/std
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.2023, 0.1994, 0.2010)
        )
    ])
    elif aug_strength == "strong":
         train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding = 4), #data augmentation
        transforms.RandomHorizontalFlip(), #randomly flips images 
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize( #normalizes data, Xnormalized = (x - mean)/std
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.2023, 0.1994, 0.2010)
        )
    ])
    else:
        raise ValueError(f"Unknown aug_strength: {aug_strength}")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.2023, 0.1994, 0.2010)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root = "data", 
        train = True, #selects split
        download = True,
        transform = train_transform #preprocessing pipeline
    )

    test_dataset = datasets.CIFAR10(
        root = "data",
        train = False,
        download = True,
        transform = test_transform
    )

    train_dataset = Subset(train_dataset, range(5000))
    test_dataset = Subset(test_dataset, range(1000))

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True, #shuffles data order in every epoch
        num_workers = num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers
    )

    return train_loader, test_loader

class SimCLRTransform:
    def __init__(self):
        self.base_tranform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            )
        ])

    def __call__(self, x):
        return self.base_tranform(x), self.base_tranform(x)
    
def get_simclr_dataloader(batch_size=256, num_workers=0):
    dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=SimCLRTransform()
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    