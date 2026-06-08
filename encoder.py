import pickle

chars = sorted(list(set(" !&()*,-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyz £ÆÉÊÔàâäæçèéêëîïòôöùûüŒœέγεθιμοςτ—‘’“”…")))

class Encoder():
    def __init__(self):
        self.chars = chars
        self.stoi = { ch:i for i, ch in enumerate(chars)}
        self.itos = { i:ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

coding = Encoder()
pickle.dump(coding, open('models/encoder_270607.pkl', 'wb'))
print(f"Encoder saved: {len(chars)} unique characters")
