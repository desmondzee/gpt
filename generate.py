import torch
import pickle
from main import BigramLanguageModel, Encoder, device, vocab_size, block_size

coding = pickle.load(open('encoder.pkl', 'rb'))

model = BigramLanguageModel()
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model = model.to(device)
model.eval()

prompt = "The "
context = torch.tensor([coding.encode(prompt)], dtype=torch.long, device=device)
with torch.no_grad():
    output = model.generate(context, max_new_tokens=1000)
print(coding.decode(output[0].tolist()))
