import torch
from torchviz import make_dot

# Reference:
# https://stackoverflow.com/questions/56816241/difference-between-detach-and-with-torch-nograd-in-pytorch

# Example 1: without .detach()
x = torch.ones(2, requires_grad=True)
y = 2 * x
z = 3 + x
r = (y + z).sum()
without_detach = make_dot(r)
without_detach.render("without_detach")

# Example 2: with .detach()
x = torch.ones(2, requires_grad=True)
y = 2 * x
# is the same as
# z = 3 + x.data()
# but .data() is the old notation
z = 3 + x.detach()
r = (y + z).sum()
with_detach = make_dot(r)  # show_attr=True
with_detach.render("with_detach")

print(x)
print(x.detach())

# Example 3: no_grad()
# torch.no_grad() is actually a class
x = torch.ones(2, requires_grad=True)
with torch.no_grad():
    y = 2 * x
print(y.requires_grad)
