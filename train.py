import argparse

if __name__=='__main__':
    parser=argparse.ArgumentParser(
        description='This is a script to train a model for selective watermarking',
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='//data3/wcr/my_project/models/opt-1.3b',
    )
    parser.add_argument(
        "--semantic_model_path",
        type=str,
        default='/data3/wcr/my_lab/hf_models/simcse-roberta-base'
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="C4"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1
    )
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=75
    )
    parser.add_argument(
        "--batch_size",
        type=int,    
        default=5
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default='adam'
    )

    parser.add_argument(
        "--weight_decay",    
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--step_size",
        type=int,
        default=50

    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0

    )


    parser.add_argument(
        "--k_len",
        type=int,
        default=6
    )
    parser.add_argument(
        "--softmax_type",
        type=str,
        default="mix_one_hot"
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--wm_gamma",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--wm_delta",
        type=float,
        default=3.0,
    )
    parser.add_argument(
            "--z_score_factor",
            type=float,
            default=0.85,

        )
    
    parser.add_argument(
            "--sim_factor",
            type=float,
            default=1.5,

        )
    parser.add_argument(
            "--wmratio_factor",
            type=float,
            default=0.4,

        )
    parser.add_argument(
            "--save_steps",
            type=int,
            default=200,

        )
    
    parser.add_argument(
            "--method",
            type=str,
            default='MGDA',

        )
    
    parser.add_argument(
            "--load_init",
            type=bool,
            default=True,

        )
    
    parser.add_argument(
            "--init_path",
            type=str,
            default="/data3/wcr/my_project/selective_watermark_based_on_semantic/ckpt/selective_network_init1.pth",

        )
    
    parser.add_argument(
            "--entropy_args",
            type=float,
            default=3,

        )
    parser.add_argument(
            "--entropy_mul",
            type=float,
            default=1,

        )
    parser.add_argument(
            "--entropy_factor",
            type=float,
            default=0.55

        )
    
    parser.add_argument(
            "--output_fix_factor",
            type=float,
            default=0.1

        )
    
    

    
    

args=parser.parse_args()

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel
from utils.selector_network import SelectorNetwork
from train_selector import train_selector
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os
import itertools
# hashkey
hash_key =  15485863
torch.manual_seed(hash_key)
torch.cuda.manual_seed(hash_key)
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2" 

def main(args):
    device = torch.device('cuda:0')
    model_path = args.model_path
    model=AutoModelForCausalLM.from_pretrained(model_path,device_map="balanced_low_0")
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    selective_network=SelectorNetwork()
    if args.load_init:
        selective_network.load_state_dict(torch.load(args.init_path,map_location="cuda:0"))
    selective_network.to(device)
    semantic_model=AutoModel.from_pretrained(args.semantic_model_path).to(device)
    semantic_model.eval()
    model.eval()
    #using the llm to inferrence ,don't need its gred
    for _,parm in model.named_parameters():
        parm.requires_grad=False
    #using the simcse to get its embedding matrix and sentence embedding,don't need grad
    for _,parm in semantic_model.named_parameters():
        parm.requires_grad=False
    
    if args.optimizer=="SGD" or args.optimizer=="sgd":
        optimizer=torch.optim.SGD(selective_network.parameters(), lr=args.learning_rate, momentum=0.9,weight_decay=args.weight_decay)
    else:
        optimizer=torch.optim.Adam(selective_network.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scaler=torch.amp.GradScaler()


    # load_datasets
    if(args.dataset_name)=="C4":
        dataset = load_dataset("json", data_files="/data3/wcr/LTW/datasets/c4_realnews/c4-train.00000-of-00512.json")
    dataset=dataset["train"]
    dataset=dataset.select(range(10000))
    def preprocess_function(examples):
       
        return tokenizer(
            examples["text"],
            padding="max_length",  
            truncation=True,      
            max_length=args.max_input_len,        
            return_attention_mask=True,  
            padding_side="left"

        )
    tokenized_dataset = dataset.map(preprocess_function, batched=True,remove_columns=dataset.column_names)
    tokenized_dataset = tokenized_dataset.with_format("torch")
    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True)

    #################
    # start  training 
    #################
    total_batches = len(dataloader)  # 训练集 batch 数
    total_steps = args.epochs * total_batches
    progress_bar = tqdm(total=total_steps, desc="Training", ncols=100, unit="steps")
    softmax_info={
        'softmax_type':args.softmax_type,
        'ratio':args.ratio,
                  }
    for epoch in range(args.epochs):
        steps=0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            output_embdding_wm,output_embdding_nwm,output_greenlist,watermarked_record,entropy_record,wmratio_record=train_selector(device,model,semantic_model,args.k_len,selective_network,input_ids,attention_mask,args.max_new_tokens,args.wm_gamma,args.wm_delta,softmax_info)

            #get sentence-level embeddding
            attention_masks=torch.ones(output_embdding_wm.shape[0],output_embdding_wm.shape[1]).to(device)
            embed_wm=semantic_model(inputs_embeds=output_embdding_wm,output_hidden_states=True, return_dict=True,attention_mask=attention_masks).pooler_output
            embed_nwm=semantic_model(inputs_embeds=output_embdding_nwm,output_hidden_states=True, return_dict=True,attention_mask=attention_masks).pooler_output

            # get  cos_sim
            cos_sim = F.cosine_similarity(embed_wm, embed_nwm, dim=1)
            
            # get detection z-score
            total_green=torch.sum(output_greenlist,dim=-1,keepdim=False)
            total_selected=torch.sum(watermarked_record,dim=-1,keepdim=False)
            z_score=(total_green-args.wm_gamma*total_selected)/torch.sqrt(total_selected*(1-args.wm_gamma)*args.wm_gamma)
            z_score=z_score
            entropy_fix=F.binary_cross_entropy(watermarked_record, F.sigmoid(args.entropy_mul*(entropy_record-args.entropy_args)))
            output_fix=-torch.mean((watermarked_record-0.5)**2)
            mseloss = nn.MSELoss() 
            wmratio_fix=mseloss(watermarked_record, 1-wmratio_record)


            # use MGDA or other moo algorithms to update the selective_network
            if args.method=="MGDA":


                sim_loss=-args.sim_factor*torch.mean(cos_sim)+args.entropy_factor*entropy_fix+args.output_fix_factor*output_fix
                z_loss= args.wmratio_factor*wmratio_fix +torch.mean(-args.z_score_factor*z_score) +args.output_fix_factor*output_fix
                vec_s=[]
                vec_z=[]
                scaler.scale(sim_loss).backward(retain_graph=True)
                for param in itertools.chain(selective_network.parameters()):
                    vec_s.append(param.grad.view(-1))
                vec_s=torch.cat(vec_s)

                optimizer.zero_grad()

                scaler.scale(z_loss).backward(retain_graph=True)
                for param in itertools.chain(selective_network.parameters()):
                    vec_z.append(param.grad.view(-1))
                vec_z=torch.cat(vec_z)

                if torch.dot(vec_z, vec_s) >= torch.dot(vec_z, vec_z):
                    factor = 1.0
                elif torch.dot(vec_z, vec_s) >= torch.dot(vec_s, vec_s):
                    factor = 0.0
                else:
                    factor = (torch.dot(vec_s - vec_z, vec_s)/torch.dot(vec_s - vec_z, vec_s - vec_z)).item()
                
                factor = max(factor, 0.001) # ensure the weight for L_D is not too low    
                vec = factor * vec_z + (1 - factor) * vec_s

                grad_position = 0
                for param in itertools.chain(selective_network.parameters()):
                    param_numel = param.numel()
                    param_grad = vec[grad_position:grad_position + param_numel]
                    param_grad = param_grad.view_as(param)
                    param.grad = param_grad
                    grad_position += param_numel

                scaler.step(optimizer)
                scaler.update()
                progress_bar.update(1)
                scheduler.step()
                steps+=1

                del vec_s,vec_z
                


            else:

                # or get the weighted loss to update the selective_network
                #simple ver
                sim_loss=-torch.mean(cos_sim)
                z_loss=-args.z_score_factor*z_score
                
                loss=sim_loss+z_loss
                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()
                progress_bar.update(1)
                scheduler.step()
                steps+=1
                
    
            
            print(f"sim_loss:{sim_loss},z_loss:{z_loss}")
            del output_embdding_wm,output_embdding_nwm,output_greenlist,watermarked_record
            torch.cuda.empty_cache()  
            if steps%args.save_steps==0 and steps!=0:
                torch.save(selective_network.state_dict(), f"/data3/wcr/LTW/ckpt/tmp/selective_network_epoch{epoch}_step{steps}.pth")
        torch.save(selective_network.state_dict(), f"/data3/wcr/LTW/ckpt/tmp/selective_network_epoch{epoch}_step{steps}.pth")    
            

        


    
    


    



main(args)

