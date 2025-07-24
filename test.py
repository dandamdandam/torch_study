import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

# 1. 데이터 정의
x = torch.randn(3, 4, requires_grad=True) 
y = torch.tensor([0, 2, 1])
print("[Input x]:")
print(x)

# 2. 모델 정의
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        self.hidden = self.fc1(x)              # hook: pre-activation
        self.activated = self.relu(self.hidden) # hook: post-activation
        out = self.fc2(self.activated)
        return out
    #def __init__(self, input_dim, hidden_dim, activation='tanh'):
        #super().__init__()
        #self.W = nn.Linear(input_dim, hidden_dim)  # W와 b 포함됨
        #self.u = nn.Parameter(torch.randn(hidden_dim))  # u는 직접 정의
#
        ## 활성화 함수 설정
        #if activation == 'tanh':
            #self.activation = torch.tanh
        #elif activation == 'relu':
            #self.activation = F.relu
        #else:
            #raise ValueError("지원하지 않는 활성화 함수입니다.")
#
    #def forward(self, x):
        #h = self.activation(self.W(x))  # h = f(Wx + b)
        #s = torch.dot(self.u, h)        # s = u^T h
        #y_hat = torch.sigmoid(s)        # 예측 확률
        #return y_hat

model = SimpleMLP()

# 중간 layer의 출력에 대해 hook을 설정하여 gradient를 출력
def save_grad(name):
    def hook_fn(grad):
        print(f"[Gradient at {name}]:\n{grad}\n")
    return hook_fn

logits = model(x)
model.hidden.retain_grad()
model.activated.retain_grad()

model.hidden.register_hook(save_grad("Hidden (Pre-Activation)"))
model.activated.register_hook(save_grad("Hidden (Post-Activation)"))

# 손실 함수 및 역전파
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, y)

# Computation graph 시각화
make_dot(loss, params=dict(model.named_parameters())).render("backprop_graph", format="png")

# Backward pass
loss.backward()

# Gradient 출력
print("[Model Parameter Gradients]")
for name, param in model.named_parameters():
    print(f"{name}.grad:\n{param.grad}\n")

# 입력 x에 대한 gradient도 추적
print("[Input x Gradient]:")
print(x.grad)

