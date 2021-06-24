import torch

def gram(x):
    b, c, h, w = x.size()
    g = torch.bmm(x.view(b, c, h * w), x.view(b, c, h * w).transpose(1, 2))
    return g.div(h * w)

def gram_loss(f1, f2):
    return ((gram(f1) - gram(f2))**2).view(f1.shape[0], -1).mean(1)

def gram_trace_loss(f1, f2):
    return ((torch.diagonal(gram(f1), dim1=1, dim2=2) - torch.diagonal(gram(f2), dim1=1, dim2=2))**2).view(f1.shape[0], -1).mean(1)