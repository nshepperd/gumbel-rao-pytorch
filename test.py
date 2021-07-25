import torch
from gumbel_rao import gumbel_rao


x = torch.tensor([[0,0,1.0,0]])
x.requires_grad_()
opt = torch.optim.Adam([x], lr=0.001)

outcomes = torch.tensor([1,2,3,4], dtype=torch.float32)

total_grad = torch.zeros_like(x)
total_var = torch.zeros_like(x)
counter = 1
while True:
    loss = (gumbel_rao(x,k=100,temp=0.01) * outcomes).square().mean()
    grad = torch.autograd.grad(loss, x)[0]

    exact_grad = torch.autograd.grad((torch.nn.functional.softmax(x, dim=-1) * outcomes).sum(), x)[0]

    total_grad.add_(grad)
    total_var.add_(grad.square())

    if counter % 500 == 0:
        print((total_grad/counter - exact_grad),
              (total_var/counter - (total_grad/counter).square()).sqrt())

    counter += 1
