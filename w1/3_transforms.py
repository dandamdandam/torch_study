import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# 데이터셋 로딩할 떄, 어떻게 학습에 적합한 상태로 변환하냐?

# ToTensor: PIL 이미지나 NumPy 배열을 PyTorch 텐서로 변환
# Lambda: 사용자 정의 함수를 적용하여 라벨을 변환 (여기선 원-핫 인코딩)
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

_, label = ds[0]
print(type(label))
print(label.shape)

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

_, label = ds[0]
print(type(label))
print(label.shape)

"""
<class 'torch.Tensor'>
torch.Size([10])
<class 'int'>
Traceback (most recent call last):  -> 라벨이 int라 에러
  File "/home/dandamdandam/python/w1/3_transforms.py", line 26, in <module>
    print(label.shape)
          ^^^^^^^^^^^
AttributeError: 'int' object has no attribute 'shape'
"""
