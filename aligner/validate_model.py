import torch


def validate_model(
        model, 
        tokenizer, 
        data,
        batch_size,
    ):
    loss_fn = torch.nn.CrossEntropyLoss()

    losses = []
    for i in range(0, len(data), batch_size):
        current_data = data[i:i+batch_size]
        mdf = [sample[0] for sample in current_data]
        ru = [sample[1] for sample in current_data]

        with torch.inference_mode():
            batch = tokenizer(ru+mdf, return_tensors='pt', padding=True, truncation=True, max_length=128).to(model.device)
            embeddings = torch.nn.functional.normalize(model.bert(**batch, output_hidden_states=True).pooler_output)

        all_scores = torch.matmul(embeddings[:batch_size], embeddings[batch_size:].T)

        loss = loss_fn(
            all_scores, torch.arange(batch_size, device=model.device)
        ) + loss_fn(
            all_scores.T, torch.arange(batch_size, device=model.device)
        )

        losses.append(loss.item())
    
    return losses
