import torch
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

print(xbow[0])