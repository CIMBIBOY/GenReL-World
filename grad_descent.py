import torch
from torch.autograd import Variable as V

# GOAL: Do gradient descent on unrolled network that is simply:
#   y = y + w*x  -->  y(t) = y(t-1) + w*x(t)

# Random training data
X = V(torch.randn(100,1).cuda())
Y = V(torch.randn(100,1).cuda())

nGD = 10        # number of gradient descent iterations
nTime = 5       # number of unrolling time steps

# Initialize things
gamma = 0.1
w = V(torch.randn(1,1).cuda(), requires_grad=True)
Yest = V((0.5*torch.ones(100,1)).cuda()) # Don't really care about values in Yest at this point, just allocating GPU memory

for iGD in range(nGD):
    # At start of processing for each GD iteration, the
    # output estimate, Yset, should be set to an initial
    # estimate of some fixed value.  E.g., ...
    Yest.data.zero_().add_(0.5)                # This line fails on second GD iteration.
    # Yest = V((0.5*torch.ones(100,1)).cuda()) # This works, but I think it's allocating GPU memory every time, and thus slow.

    for iTime in range(nTime):
        Yest = Yest + w*X
    cost = torch.mean((Y - Yest)**2)    
    cost.backward() # compute gradients
    w.data.sub_(gamma*w.grad.data) # Update parameters
    w.grad.data.zero_() # Reset gradients to zeros