import torch
from torch.distributions import Beta

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(5.0, requires_grad=True)
x = torch.tensor(0.3)

dist = Beta(a, b)
p = 1.0 - dist.cdf(x)
p.backward()

print(a.grad, b.grad)