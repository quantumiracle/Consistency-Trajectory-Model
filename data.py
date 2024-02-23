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
    elif name == 'cifar10':  # https://github.com/openai/consistency_models_cifar10/blob/d086e77dfdc30b51671685f9eec90c90d4f4eaa6/jcm/datasets.py#L90
        transform = torchvision.transforms.Compose([
            data_augmentation,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return torchvision.datasets.CIFAR10(root='tmp', train=train, download=True, transform=transform)
    elif name == 'imagenet64':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(64),  # Resize the image to 64x64 pixels
            # torchvision.transforms.CenterCrop(224),  # Crop the image to 224x224 pixels around the center
            torchvision.transforms.ToTensor(),  # Convert the image to a tensor
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the image
                                std=[0.229, 0.224, 0.225])
        ])
        return torchvision.datasets.ImageNet(root='tmp', train=train, download=True, transform=transform)
    
    
    raise ValueError(f'Dataset {name} not found')