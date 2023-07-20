import torchvision.transforms as transforms
import torchvision.datasets as datasets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images to [-1, 1]
])

train_dataset = datasets.CIFAR10(root='path/to/cifar10', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='path/to/cifar10', train=False, download=True, transform=transform)