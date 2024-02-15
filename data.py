import torchvision

def get_dataset(name, train, evaluation=False):
    def data_augmentation(x):
        if not evaluation:
            x = torchvision.transforms.RandomHorizontalFlip()(x)
        return x
    if name == 'fmnist':
        transform = torchvision.transforms.Compose([
            data_augmentation,
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5)),
        ])
        return torchvision.datasets.FashionMNIST(root='tmp', train=train, download=True, transform=transform)
    elif name == 'mnist':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5)),
        ])
        return torchvision.datasets.MNIST(root='tmp', train=train, download=True, transform=transform)
    elif name == 'cifar10':
        transform = torchvision.transforms.Compose([
            data_augmentation,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5)),
        ])
        return torchvision.datasets.CIFAR10(root='tmp', train=train, download=True, transform=transform)
    
    
    raise ValueError(f'Dataset {name} not found')