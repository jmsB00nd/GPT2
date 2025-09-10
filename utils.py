import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def text_to_tokens_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def tokens_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch , target_batch , model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss


def calc_loss_loader(dataloader, model, device, num_batchs=None):
    total_loss = 0
    if len(dataloader) == 0 :
        return float("nan")
    elif num_batchs is None :
        num_batchs = len(dataloader)
    else :
        num_batchs = min(num_batchs, len(dataloader))
    
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batchs :
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else :
            break
    
    return total_loss / num_batchs


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batchs=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batchs=eval_iter)
    
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_tokens_ids(start_context, tokenizer)
    with torch.no_grad():
        tokens_ids = generate_text_simple(model, encoded, 50, context_size)
    decoded = tokens_ids_to_text(tokens_ids, tokenizer)
    print(decoded.replace("\n", " "))
    model.train()


#A modified text generation function with more diversity
def generate(model, idx, max_new_tokens, context_size,temperature=0.0, top_k=None, eos_id=None):
    for _ in max_new_tokens :
        idx_cond = idx[:, -context_size, :]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1 , :]

        if top_k is not None:
            top_logits , _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else :
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        
        if idx_next == eos_id :
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx

