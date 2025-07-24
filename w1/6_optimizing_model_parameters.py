"""
각 iteration에서

1. output에 대한 추정을 하고
2. loss(에러)를 계산하고
3. error에 대한 미분값을 수정한 후
4. 파라미터 최적화
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# load data

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# define model

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Initialize the model

model = NeuralNetwork()

learning_rate = 1e-3
batch_size = 64
epochs = 5

loss_fn = nn.CrossEntropyLoss()

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # forward
        pred = model(X)
        loss = loss_fn(pred, y)

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 손실 함수와 옵티마이저 정의
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

"""
Epoch 1
-------------------------------
loss: 2.302167  [   64/60000]
loss: 2.291215  [ 6464/60000]
loss: 2.272437  [12864/60000]
loss: 2.266811  [19264/60000]
loss: 2.248005  [25664/60000]
loss: 2.225664  [32064/60000]
loss: 2.220840  [38464/60000]
loss: 2.190839  [44864/60000]
loss: 2.187865  [51264/60000]
loss: 2.155950  [57664/60000]
Test Error: 
 Accuracy: 39.7%, Avg loss: 2.152049 

Epoch 2
-------------------------------
loss: 2.160172  [   64/60000]
loss: 2.150293  [ 6464/60000]
loss: 2.094628  [12864/60000]
loss: 2.109003  [19264/60000]
loss: 2.051951  [25664/60000]
loss: 2.004755  [32064/60000]
loss: 2.011121  [38464/60000]
loss: 1.936912  [44864/60000]
loss: 1.940608  [51264/60000]
loss: 1.867333  [57664/60000]
Test Error: 
 Accuracy: 47.0%, Avg loss: 1.871476 

Epoch 3
-------------------------------
loss: 1.902939  [   64/60000]
loss: 1.872132  [ 6464/60000]
loss: 1.762636  [12864/60000]
loss: 1.798253  [19264/60000]
loss: 1.684732  [25664/60000]
loss: 1.652992  [32064/60000]
loss: 1.651889  [38464/60000]
loss: 1.567555  [44864/60000]
loss: 1.588994  [51264/60000]
loss: 1.486039  [57664/60000]
Test Error: 
 Accuracy: 57.0%, Avg loss: 1.510673 

Epoch 4
-------------------------------
loss: 1.575957  [   64/60000]
loss: 1.540493  [ 6464/60000]
loss: 1.404015  [12864/60000]
loss: 1.465514  [19264/60000]
loss: 1.352163  [25664/60000]
loss: 1.358525  [32064/60000]
loss: 1.354204  [38464/60000]
loss: 1.293796  [44864/60000]
loss: 1.321643  [51264/60000]
loss: 1.225413  [57664/60000]
Test Error: 
 Accuracy: 62.8%, Avg loss: 1.253828 

Epoch 5
-------------------------------
loss: 1.330732  [   64/60000]
loss: 1.306882  [ 6464/60000]
loss: 1.155864  [12864/60000]
loss: 1.249610  [19264/60000]
loss: 1.130356  [25664/60000]
loss: 1.161045  [32064/60000]
loss: 1.167558  [38464/60000]
loss: 1.118736  [44864/60000]
loss: 1.151682  [51264/60000]
loss: 1.068609  [57664/60000]
Test Error: 
 Accuracy: 64.8%, Avg loss: 1.089559 

Epoch 6
-------------------------------
loss: 1.162557  [   64/60000]
loss: 1.155196  [ 6464/60000]
loss: 0.987333  [12864/60000]
loss: 1.111035  [19264/60000]
loss: 0.987877  [25664/60000]
loss: 1.023914  [32064/60000]
loss: 1.048584  [38464/60000]
loss: 1.004449  [44864/60000]
loss: 1.037020  [51264/60000]
loss: 0.968161  [57664/60000]
Test Error: 
 Accuracy: 66.5%, Avg loss: 0.981049 

Epoch 7
-------------------------------
loss: 1.042913  [   64/60000]
loss: 1.055196  [ 6464/60000]
loss: 0.869819  [12864/60000]
loss: 1.017459  [19264/60000]
loss: 0.895416  [25664/60000]
loss: 0.926095  [32064/60000]
loss: 0.970148  [38464/60000]
loss: 0.929340  [44864/60000]
loss: 0.956451  [51264/60000]
loss: 0.900711  [57664/60000]
Test Error: 
 Accuracy: 67.8%, Avg loss: 0.906717 

Epoch 8
-------------------------------
loss: 0.954293  [   64/60000]
loss: 0.985714  [ 6464/60000]
loss: 0.785339  [12864/60000]
loss: 0.951066  [19264/60000]
loss: 0.833485  [25664/60000]
loss: 0.854433  [32064/60000]
loss: 0.915086  [38464/60000]
loss: 0.878841  [44864/60000]
loss: 0.898048  [51264/60000]
loss: 0.852604  [57664/60000]
Test Error: 
 Accuracy: 69.1%, Avg loss: 0.853450 

Epoch 9
-------------------------------
loss: 0.885928  [   64/60000]
loss: 0.933861  [ 6464/60000]
loss: 0.722413  [12864/60000]
loss: 0.901810  [19264/60000]
loss: 0.789686  [25664/60000]
loss: 0.800671  [32064/60000]
loss: 0.873671  [38464/60000]
loss: 0.843637  [44864/60000]
loss: 0.854169  [51264/60000]
loss: 0.816094  [57664/60000]
Test Error: 
 Accuracy: 70.4%, Avg loss: 0.813401 

Epoch 10
-------------------------------
loss: 0.831081  [   64/60000]
loss: 0.892572  [ 6464/60000]
loss: 0.673913  [12864/60000]
loss: 0.863847  [19264/60000]
loss: 0.756934  [25664/60000]
loss: 0.759474  [32064/60000]
loss: 0.840297  [38464/60000]
loss: 0.817702  [44864/60000]
loss: 0.820046  [51264/60000]
loss: 0.786786  [57664/60000]
Test Error: 
 Accuracy: 71.5%, Avg loss: 0.781781 

Done!
"""
