import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import numpy as np
from data import get_dataset

# Function to load CIFAR-10
def load_cifar10(batch_size=32):
    trainset = get_dataset('cifar10', train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    return trainloader

# Function to extract features using InceptionV3
def extract_features(dataloader):
    # Load pretrained InceptionV3 model
    model = inception_v3(pretrained=True)
    model.eval()  # Set model to evaluation mode
    model.fc = torch.nn.Identity()  # Modify the model to output features before classification layer
    
    features = []
    with torch.no_grad():  # No need to calculate gradients
        for data in dataloader:
            images, _ = data
            output = model(images)  # Extract features
            features.append(output.cpu().numpy())
    
    features = np.concatenate(features, axis=0)  # Combine batched features
    return features

# Calculate mean and covariance of features
def calculate_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

# Main workflow
if __name__ == "__main__":
    dataloader = load_cifar10(batch_size=64)  # Load CIFAR-10
    features = extract_features(dataloader)  # Extract features
    mu_ref, sigma_ref = calculate_statistics(features)  # Calculate statistics

    # Save the reference statistics
    np.savez('author_ckpt/cifar10_ref_stats.npz', mu=mu_ref, sigma=sigma_ref)

    print("Reference mean and covariance matrix saved.")
