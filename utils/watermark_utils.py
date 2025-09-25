import torch



def get_green_id(vocab_size,device,gamma,prev_token):
    hash_key = 15485863
    rng=torch.Generator(device=device)
    rng.manual_seed(hash_key * prev_token)
    greenlist_size = int(vocab_size * gamma)
    vocab_permutation = torch.randperm(vocab_size, device=device, generator=rng)

    # get green_ids
    greenlist_ids = vocab_permutation[:greenlist_size]
    
    return greenlist_ids
   

#实现kgw算法中的获取
def get_batched_green_mask(scores,gamma,prev_ids): 
    vocab_size = scores.shape[-1]
    batch=scores.shape[0]
    device = scores.device
    batched_greenid=[ None for i in range(batch)]
    for b in range(batch):
        batched_greenid[b]=get_green_id(vocab_size,device,gamma,prev_ids[b].item()) 
    batched_greenmask=torch.zeros_like(scores)
    for b in range(batch):
        batched_greenmask[b][batched_greenid[b]]=1
    batched_greenmask=batched_greenmask.bool()
    return batched_greenmask
    

     

      
def get_green_id_unigram(vocab_size,device,gamma):
    hash_key = 15485863
    rng=torch.Generator(device=device)
    rng.manual_seed(hash_key )
    greenlist_size = int(vocab_size * gamma)
    vocab_permutation = torch.randperm(vocab_size, device=device, generator=rng)

    # get green_ids
    greenlist_ids = vocab_permutation[:greenlist_size]
    
    return greenlist_ids
   

#嵌入unigram的水印方式（kgw-0）
def get_batched_green_mask_unigram(scores,gamma): 
    vocab_size = scores.shape[-1]
    batch=scores.shape[0]
    device = scores.device
    batched_greenid=[ None for i in range(batch)]
    for b in range(batch):
        batched_greenid[b]=get_green_id_unigram(vocab_size,device,gamma) 
    batched_greenmask=torch.zeros_like(scores)
    for b in range(batch):
        batched_greenmask[b][batched_greenid[b]]=1
    batched_greenmask=batched_greenmask.bool()
    return batched_greenmask