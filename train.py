from utils import evaluate_model, generate_and_print_sample, calc_loss_batch
import torch


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):

    train_losses, val_losses, token_seen_track = [], [], []
    tokens_seen , global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch , target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1
            tokens_seen += input_batch.numlel()

            if global_step % eval_freq :
                train_loss, val_loss =  evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                token_seen_track.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "f"Train loss {train_loss:.3f}, "f"Val loss {val_loss:.3f}")

        
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, token_seen_track
