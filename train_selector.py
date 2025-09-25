from utils.selector_network import SelectorNetwork
from utils.watermark_utils import get_batched_green_mask
import torch
import torch.nn.functional as F
from transformers import AutoModel, OPTForCausalLM, AutoTokenizer,AutoModelForCausalLM





def train_selector(device,model,semantic_model,k,selector,input_ids,attention_mask,max_len,gamma,dalta,softmax_info):
    
    batch_size = input_ids.shape[0]

    input_ids_nwm=input_ids
    input_ids_wm=input_ids.clone()
    watermarked_record=torch.empty((batch_size,0),dtype=torch.float32).to(device)
    entropy_record=torch.empty((batch_size,0),dtype=torch.float32).to(device)
    wmratio_record=torch.empty((batch_size,0),dtype=torch.float32).to(device)
    
    output_greenlist=torch.empty((batch_size,0),dtype=torch.float32).to(device)

    embedding_matrix = semantic_model.get_input_embeddings().weight
    semvocab_size = embedding_matrix.shape[0]


    output_embdding_wm=torch.empty((batch_size,0,embedding_matrix.shape[1]),dtype=torch.float32).to(device)
    output_embdding_nwm=torch.empty((batch_size,0,embedding_matrix.shape[1]),dtype=torch.float32).to(device)

    input_embedding = semantic_model.get_input_embeddings()(input_ids_wm).to(device)
    sem_embed=semantic_model(inputs_embeds=input_embedding[:,-k:,:], attention_mask=attention_mask[:,-k:], output_hidden_states=True, return_dict=True).pooler_output
    
    
    for _ in range(max_len):
        # first generate the watermarked text
        with torch.no_grad():
            input_ids_wm1=input_ids_wm.to(model.device)
            attention_mask1=attention_mask.to(model.device)
            logits = model(input_ids=input_ids_wm1, attention_mask=attention_mask1).logits
            logits=logits.to(device)
        # the new new logits generated
        new_logits=logits[:, -1, :].squeeze(dim=1)
        #calculate the   Shannon  entropy of the new logits(following the work of SWEET))
        raw_probs = F.softmax(new_logits, dim=-1)
        entropy = -torch.where(raw_probs > 0, raw_probs * raw_probs.log(), raw_probs.new([0.0])).sum(dim=-1)
        entropy = entropy.unsqueeze(dim=-1).to(device)
        entropy_record = torch.cat((entropy_record, entropy), dim=-1)
        

        # calculate the watermarked ratio
        if watermarked_record.shape[-1] == 0:
            watered_ratio = torch.full((watermarked_record.shape[0], 1), float(0)).to(device)
        else:
            watered_ratio = watermarked_record.mean(dim=-1, keepdim=True).to(device) 
            
        wmratio_record= torch.cat((wmratio_record, watered_ratio), dim=-1)
        # using the semantic embedding info to deside whether to add the watermark or not
        network_input=torch.cat((entropy, watered_ratio, sem_embed), dim=-1)

        
        watermark_mask=selector(network_input)

        #add sigmoid to make it appear to be a mask


        watermarked_record = torch.cat((watermarked_record, watermark_mask), dim=-1)
        # watermark scheme (can use other watermarking scheme)
        # get green mask 
        green_mask=get_batched_green_mask(new_logits,gamma, input_ids_wm[:, -1])
        #combine those two together to form a selective watermark's mask
        watermark_mask.expand(-1,green_mask.shape[-1])
        result_mask=watermark_mask*green_mask 
             
        new_logits=new_logits+dalta*result_mask

        # compute softmax to get the next token(using softmax only)
        #use softmax or gumbel softmax 
        softmax_type=softmax_info['softmax_type']
       
        if softmax_type=="mix_one_hot":
            next_token_softmax1 = F.softmax(new_logits, dim=-1).to(device)
            max_val, max_index = torch.max(next_token_softmax1, dim=-1, keepdim=True)
            hard_one_hot = torch.zeros_like(next_token_softmax1)
            hard_one_hot[next_token_softmax1 == max_val] = 1
            ratio=softmax_info.get("ratio",0.5) 
            next_token_softmax=next_token_softmax1*ratio+hard_one_hot*(1-ratio)
        else:
            next_token_softmax = F.softmax(new_logits, dim=-1).to(device)


        max_val, max_index = torch.max(next_token_softmax, dim=-1, keepdim=True)
        input_ids_wm = torch.cat((input_ids_wm, max_index), dim=-1)
        is_green=torch.sum(result_mask*next_token_softmax,dim=-1).unsqueeze(-1)
        output_greenlist=torch.cat((output_greenlist, is_green), dim=-1)

        # need to use embedding to get grad for computing because input_id can't

        #softmax embedding,the result will be used for cal cossim and do backprop
        
        #opt模型和选用的semantic_model（simcse-roberta-base）的embedding 0维中前面到50265之前的都对的上
        next_embedding=torch.matmul(next_token_softmax[:,:semvocab_size],embedding_matrix).unsqueeze(1)
        output_embdding_wm=torch.cat((output_embdding_wm, next_embedding), dim=1)
       #update the texts and semantic embedding for the next iteration
        input_embedding = torch.cat((input_embedding, next_embedding), dim=1)


        ## generate the non-watermarked text
        with torch.no_grad():
            input_ids_nwm1=input_ids_nwm.to(model.device)
            attention_mask1=attention_mask.to(model.device)
            logits = model(input_ids=input_ids_nwm1, attention_mask=attention_mask1).logits 
            logits=logits.to(device)
        nw_new_logits=logits[:, -1, :].squeeze(dim=1)

        softmax_type=softmax_info['softmax_type']    
        if softmax_type=="mix_one_hot":
            next_token_softmax1 = F.softmax(nw_new_logits, dim=-1).to(device)
            max_val, max_index = torch.max(next_token_softmax1, dim=-1, keepdim=True)
            hard_one_hot = torch.zeros_like(next_token_softmax1)
            hard_one_hot[next_token_softmax1 == max_val] = 1
            ratio=softmax_info.get("ratio",0.5) 
            nw_next_token_softmax=next_token_softmax1*ratio+hard_one_hot*(1-ratio)
        else:
            nw_next_token_softmax = F.softmax(nw_new_logits, dim=-1).to(device)

        nw_max_val, nw_max_index = torch.max(nw_next_token_softmax, dim=-1, keepdim=True)
        input_ids_nwm = torch.cat((input_ids_nwm, nw_max_index), dim=-1)
        nw_next_embedding=torch.matmul(nw_next_token_softmax[:,:semvocab_size],embedding_matrix).unsqueeze(1)
        output_embdding_nwm=torch.cat((output_embdding_nwm, nw_next_embedding), dim=1)


        # update the attention mask for the next iteration
        attention_mask = torch.cat((attention_mask, torch.ones(batch_size, 1).cuda()), dim=1)
        sem_embed=semantic_model(inputs_embeds=input_embedding[:,-k:,:], attention_mask=attention_mask[:,-k:], output_hidden_states=True, return_dict=True).pooler_output
    # print(watered_ratio)

 
    return output_embdding_wm,output_embdding_nwm,output_greenlist,watermarked_record,entropy_record,wmratio_record

       


        

        

        
         

        
        


        




    
    

