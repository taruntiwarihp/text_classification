import torch

def my_collate(batches):
    # return batches
    return [{key: torch.stack(value) for key, value in batch.items()} for batch in batches]

# [{key: torch.stack(value) for key, value in batch.items()} for batch in batches]

def prepare_batch_for_robert_model(batch, device='cuda', fine_tune=True):
    
    ids = batch['ids'].squeeze(1).to(device, dtype=torch.long)
    mask = batch['mask'].squeeze(1).to(device, dtype=torch.long)
    token_type_ids = batch['token_type_ids'].squeeze(1).to(device, dtype=torch.long)
    targets = batch['labels'].to(device, dtype=torch.long)

    return ids, mask, token_type_ids, targets, lengt


def prepare_batch_for_bert_model(batch, device='cuda:0'):

    # print(batch)
    # print(targets.shape)
    ids = batch['ids'].squeeze(1).to(device, dtype=torch.long)
    mask = batch['mask'].squeeze(1).to(device, dtype=torch.long)
    token_type_ids = batch['token_type_ids'].squeeze(1).to(device, dtype=torch.long)
    targets = batch['labels'].to(device, dtype=torch.long)
    # print(targets, targets.shape)    
    return ids, mask, token_type_ids, targets

def prepare_batch_for_lstm_model(batch, device='cuda'):

    ids = batch[0].to(device, dtype=torch.long)
    seq_lens = batch[2]
    targets = batch[1].to(device, dtype=torch.long)

    return ids, seq_lens, targets