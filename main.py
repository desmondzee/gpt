import urllib.request
from bs4 import BeautifulSoup
import torch

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

block_size = 8

#training parameters x,y

torch.manual_seed(888)
batch_size = 8
block_size = 8

def get_batch(split):
    get_batch_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(get_batch_data) - block_size, (batch_size,)) #rand positions in data
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print(xb.shape, xb, yb.shape, yb)

for b in range(batch_size): #batch dim
    for t in range(block_size): #time dim
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f'for{context.tolist()}, want {target}')





