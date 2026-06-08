import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(888)

B,T,C = 8,8,94
x = torch.randn(B,T,C)
print(x.shape)

#x[B,T] = mean_{i<=T}(x[b,i])
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1]
        xbow[b,t] = torch.mean(xprev, 0)

wei = torch.tril(torch.ones(T, T)) #weighted aggregation by T, T array, matrix multiply
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B, T, T) @ (B, T, C) -> (B, T, C)
print(torch.allclose(xbow, xbow2))

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x) #B,T,16
q = query(x) #B,T,16
wei = q @ k.transpose(-2, -1) #B,T,16 @ B,16,T --> B,T,T affinities

#using softmax
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf')) #all zeroes go to -infinity
wei = F.softmax(wei, dim=-1) #softmax on every row, normalisation -> exp / sum, produces weighted mask
v = value(x)
out = wei @ v
print(out[:2])