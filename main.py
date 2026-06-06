import urllib.request
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(888)
batch_size = 8
block_size = 8
max_iters = 10000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

url = "https://www.gutenberg.org/cache/epub/1260/pg1260-images.html"
with urllib.request.urlopen(url) as response:
    html = response.read()

soup = BeautifulSoup(html, 'html.parser')
text = soup.get_text()

start_marker = "START OF THE PROJECT GUTENBERG EBOOK"
end_marker = "END OF THE PROJECT GUTENBERG EBOOK"

start_idx = text.find(start_marker)
end_idx = text.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("no start/end marker")
    exit(0)

start_idx = text.find('\n', start_idx) + 1
book_text = text[start_idx:end_idx].strip()
#print(len(book_text))
#print(book_text[-1000:])

chars = sorted(list(set(book_text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

#lookup table to token mappings
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#print(encode("hi theressss"))
#print(decode(encode("hi theressss")))

data = torch.tensor(encode(book_text), dtype=torch.long)
print(data.shape, data.dtype)
#print(data[:1000])

#train/val
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#training parameters x,y -> data loading
def get_batch(split):
    get_batch_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(get_batch_data) - block_size, (batch_size,)) #rand positions in data
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() #tell pytorch not intending back prop
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

"""
xb, yb = get_batch('train')
print(xb.shape, xb, yb.shape, yb)

for b in range(batch_size): #batch dim
    for t in range(block_size): #time dim
        context = xb[b, :t+1]
        target = yb[b, t]
        #print(f'for{context.tolist()}, want {target}')
"""

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        #idx, targets are (b,T) tensor of ints
        logits = self.token_embedding_table(idx) #B,T,C

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #2D stretch for cross entropy pytorch
            targets = targets.view(B*T) #1D stretch
            loss = F.cross_entropy(logits, targets) #accuracy wrt target

        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is B,T array of indices
        for _ in range(max_new_tokens):
            logits, loss = self(idx) #new predictions
            logits = logits[:, -1, :] #last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) #distribution sampling -> B,1
            idx = torch.cat((idx, idx_next), dim=1) # B, T+1
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

#pytorch optimiser object
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for iter in range(max_iters):
    #eval loss on train/val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'iter {iter}, loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}')

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


