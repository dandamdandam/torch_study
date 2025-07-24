import torch

# layer 구성
# z = Wx + b
# loss = cross_entropy(z, y)
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True) # weight tensor - grad를 계산해야하면 true
b = torch.randn(3, requires_grad=True) # bias tensor
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# gradient 계산
loss.backward()
print(w.grad)
print(b.grad)

"""
Gradient function for z = <AddBackward0 object at 0xffff85c378e0>
Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0xffff85c378e0>
tensor([[0.0532, 0.2969, 0.1099],
        [0.0532, 0.2969, 0.1099],
        [0.0532, 0.2969, 0.1099],
        [0.0532, 0.2969, 0.1099],
        [0.0532, 0.2969, 0.1099]])
tensor([0.0532, 0.2969, 0.1099])
"""

# 특정 layer의 gradient 추적 비활성화
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

"""
True
False
False
"""

# gradients 와 Jacobian 곱

inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")

"""
First call
tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.]])

Second call
tensor([[8., 4., 4., 4., 4.],
        [4., 8., 4., 4., 4.],
        [4., 4., 8., 4., 4.],
        [4., 4., 4., 8., 4.]])

Call after zeroing gradients
tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.]])
"""