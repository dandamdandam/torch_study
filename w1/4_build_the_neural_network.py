import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 사용할 device 설정
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device") # -> Using cpu device

class NeuralNetwork(nn.Module):
    def __init__(self): # layer 정의
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x): # layer에 데이터를 어떻게 전달할지 정의
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 아무데이터로 모델 돌리기

model = NeuralNetwork().to(device)
print(model)
"""
NeuralNetwork(
    (flatten): Flatten(start_dim=1, end_dim=-1)
    (linear_relu_stack): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=10, bias=True)
    )
)
"""

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}") # -> Predicted class: tensor([8])

# 각 layer 설명

# Flatten
input_image = torch.rand(3,28,28)
print(input_image.size()) # -> torch.Size([3, 28, 28])

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size()) # -> torch.Size([3, 784])

# nn.Linear
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size()) # -> torch.Size([3, 20])

# nn.ReLU
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
"""
Before ReLU: tensor([[ 0.1951, -0.0619,  0.2939, -0.5349, -0.0036,  0.4273, -0.1554, -0.2866,
         -0.2417, -0.1175,  0.6333,  0.2363,  0.2904, -0.4140,  0.2301,  0.4521,
         -0.0210, -0.0065,  0.0769,  0.0531],
        [-0.1304, -0.2811,  0.5034, -0.5930,  0.1781,  0.5326,  0.1076, -0.3330,
         -0.0190, -0.1137,  0.3873,  0.1991, -0.0246, -0.9059,  0.5892,  0.2267,
         -0.1552,  0.6670,  0.2351, -0.1658],
        [-0.2612, -0.0465,  0.1253, -0.5538,  0.1725,  0.2279,  0.1072, -0.1965,
         -0.4236, -0.5104,  0.7570,  0.2923,  0.1224, -0.3986,  0.6310, -0.0074,
         -0.1664,  0.3070,  0.1847, -0.1143]], grad_fn=<AddmmBackward0>)


After ReLU: tensor([[0.1951, 0.0000, 0.2939, 0.0000, 0.0000, 0.4273, 0.0000, 0.0000, 0.0000,
         0.0000, 0.6333, 0.2363, 0.2904, 0.0000, 0.2301, 0.4521, 0.0000, 0.0000,
         0.0769, 0.0531],
        [0.0000, 0.0000, 0.5034, 0.0000, 0.1781, 0.5326, 0.1076, 0.0000, 0.0000,
         0.0000, 0.3873, 0.1991, 0.0000, 0.0000, 0.5892, 0.2267, 0.0000, 0.6670,
         0.2351, 0.0000],
        [0.0000, 0.0000, 0.1253, 0.0000, 0.1725, 0.2279, 0.1072, 0.0000, 0.0000,
         0.0000, 0.7570, 0.2923, 0.1224, 0.0000, 0.6310, 0.0000, 0.0000, 0.3070,
         0.1847, 0.0000]], grad_fn=<ReluBackward0>)
"""

# nn.Sequential
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
print(f"Logits: {logits}\n\n")
"""
Logits: tensor([[-0.1819, -0.1919,  0.1379, -0.3846,  0.2632,  0.3739,  0.2306,  0.1726,
         -0.0384,  0.2945],
        [-0.0020, -0.2772,  0.1335, -0.2071,  0.2341,  0.2459,  0.2624, -0.0256,
          0.1044,  0.4339],
        [-0.1364, -0.2736,  0.0198, -0.3491,  0.2011,  0.3636,  0.3916,  0.0203,
          0.2264,  0.3203]], grad_fn=<AddmmBackward0>)
"""

# nn.Softmax

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(f"Predicted probabilities: {pred_probab}\n\n")
"""
Predicted probabilities: tensor([[0.0758, 0.0751, 0.1044, 0.0619, 0.1183, 0.1322, 0.1145, 0.1081, 0.0875,
         0.1221],
        [0.0893, 0.0678, 0.1022, 0.0727, 0.1130, 0.1144, 0.1163, 0.0872, 0.0993,
         0.1380],
        [0.0782, 0.0682, 0.0915, 0.0633, 0.1097, 0.1290, 0.1327, 0.0915, 0.1125,
         0.1235]], grad_fn=<SoftmaxBackward0>)
"""


# Model Parameters

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
"""
Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[-0.0021, -0.0006, -0.0038,  ...,  0.0041, -0.0027,  0.0353],
        [ 0.0278,  0.0106, -0.0282,  ..., -0.0035,  0.0285,  0.0256]],
       grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0314,  0.0155], grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[ 0.0370,  0.0380, -0.0373,  ..., -0.0412, -0.0202,  0.0152],
        [ 0.0289,  0.0123,  0.0233,  ..., -0.0149,  0.0315, -0.0412]],
       grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([-0.0209,  0.0042], grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[ 0.0411,  0.0259,  0.0366,  ..., -0.0306, -0.0197, -0.0419],
        [-0.0014,  0.0405,  0.0249,  ..., -0.0246,  0.0434,  0.0236]],
       grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([-0.0339, -0.0208], grad_fn=<SliceBackward0>) 
"""
