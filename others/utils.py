import torch 


a = torch.Tensor([[1,2],[3,4],[5,6]])
print(a.shape)
b = torch.unsqueeze(a, dim=1)
print(b.shape)
print(b)
c = torch.unsqueeze(a, dim=0)
print(c.shape)
print(c)
print(torch.sum((b-c)**2,dim=-1)**0.5)