import torch
from GPT2 import GPTModel
from utils import generate_text_simple
import tiktoken
from tokenizer.DataLoader import create_dataloader_v1
from utils import calc_loss_loader

torch.manual_seed(123)

GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "context_length" : 256,
    "emb_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False,
}

file_path = "/home/jmsb00nd/Documents/NLP/Building LLM's (from scrtach)/dataset/the-verdict.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text_data = f.read()

train_ratio = 0.9
split_idx = int(len(text_data) * train_ratio)
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


train_loader = create_dataloader_v1(train_data, 2, GPT_CONFIG_124M["context_length"], GPT_CONFIG_124M["context_length"], True, True, 0)
val_loader = create_dataloader_v1(val_data, 2, GPT_CONFIG_124M["context_length"], GPT_CONFIG_124M["context_length"], False, False, 0)


print("Train data : ")
for x, y in train_loader:
    print(x.shape, y.shape)

print("Val data : ")
for x, y in val_loader:
    print(x.shape, y.shape)


model = GPTModel(GPT_CONFIG_124M)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
    
print("Training loss:", train_loss)
print("Validation loss:", val_loss)


# load and save 
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)

torch.save({
"model_state_dict": model.state_dict(),
"optimizer_state_dict": optimizer.state_dict(),
},
"model_and_optimizer.pth"
)

model.load_state_dict(torch.load("model.pth", map_location=device)) 
model.eval()

checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train();