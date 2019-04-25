import torch
from torch import Tensor
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad

# some toy data
x = Variable(Tensor([[4., 2.]]), requires_grad=False)
y = Variable(Tensor([1.]), requires_grad=False)

# linear model and squared difference loss
model = nn.Linear(2, 1)
loss = torch.sum((y - model(x)) ** 2)

# instead of using loss.backward(), use torch.autograd.grad() to compute gradients
loss_grads = grad(loss, model.parameters(), create_graph=True)

dW, db = loss_grads
W, b = tuple(model.parameters())

# compute the second order derivative w.r.t. each parameter
d2loss = []
for param, grd in zip(model.parameters(), loss_grads):
    # for idx in iterator_over_tensor(param):
    drv = grad(grd, param, create_graph=True, retain_graph=True)
    d2loss.append(drv)
    print(param, drv)
