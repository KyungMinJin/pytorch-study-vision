import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from utils.model.ResNet import ResNet18
import argparse
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='cifar10 cls model')
parser.add_argument('--lr', default=0.1)
parser.add_argument('--resume', default=None)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--batch_size_test', default=100)
parser.add_argument('--num_worker', default=4)
parser.add_argument('--logdir', default='logs')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

dataset_train = CIFAR10(root='./data', train=True, download=False, transform=transforms_train)
dataset_test = CIFAR10(root='./data', train=False, download=False, transform=transforms_test)

train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_worker)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Making model..')

model = ResNet18().to(device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('The number of parameters of model is', num_params)

if args.resume is not None:
    checkpoints = torch.load('./checkpoints/' + args.resume)
    model.load_state_dict(checkpoints['model'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

decay_epoch = [32000, 48000]
step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epoch, gamma=0.1)
writer = SummaryWriter(args.logdir)

def train(epoch, global_steps):
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        global_steps += 1
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_lr_scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100 * correct / total
    print(f'train epoch : {epoch} [{batch_idx+1} / {len(train_loader)}] | loss: {train_loss/ (batch_idx + 1):.3f} | acc: {acc:.3f}')

    writer.add_scalar('log/train error', 100 - acc, global_steps)
    return global_steps

def test(epoch, best_acc, global_steps):
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        acc = 100 * correct / total
        print(
            f'test epoch : {epoch} [{batch_idx+1} / {len(test_loader)}] | loss: {test_loss / (batch_idx + 1):.3f} | acc: {acc:.3f}')

        writer.add_scalar('log/test error', 100 - acc, global_steps)

    if acc > best_acc:
        print('==> Saving model')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, './checkpoints/ckpt.pth')
        best_acc = acc

    return best_acc

if __name__ == '__main__':
    best_acc = 0
    epoch = 0
    global_steps = 0

    if args.resume is not None:
        test(epoch=0, best_acc=0)
    else:
        while True:
            epoch += 1
            global_steps = train(epoch, global_steps)
            best_acc = test(epoch, best_acc, global_steps)
            print('best test accuracy is', best_acc)

            if global_steps >= 64000:
                break