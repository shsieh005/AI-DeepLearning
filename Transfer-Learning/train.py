import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-4  # 0.0001

transform = transforms.Compose(
    [transforms.Resize([224, 224]),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR10(root='path/to/cifar10', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='path/to/cifar10', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

resnet50 = torchvision.models.resnet50(weights="IMAGENET1K_V2")
print(resnet50)

resnet50.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
resnet50 = resnet50.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet50.parameters(), lr=learning_rate)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grcount_parameters())



def train(epoch):
    resnet50.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = resnet50(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    train_accuracy = 100. * correct / total
    print(f"Epoch {epoch}, Training Accuracy: {train_accuracy:.2f}%")


def test():
    resnet50.eval()
    test_loss = 0
    correct = 0
    n_samples = 0
    n_correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            output = resnet50(data)
            test_loss += criterion(output, labels).item()
            _, predicted = torch.max(output, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    acc = 100.0 * n_correct / n_samples
    print(f"Test Loss: {test_loss}, Accuracy: {acc}%")


for epoch in range(2):
    train(epoch)
    test()

torch.save(resnet50.state_dict(), "../../ResNet-PyTorch/ResNet/resnet50_cifar10.pth")
