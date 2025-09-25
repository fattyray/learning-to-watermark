# %%
from datasets import load_dataset, Dataset
from watermark import Watermark
import torch
from transformers import AutoTokenizer
from watermark import Detector
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModel
import torch
import torch.nn.functional as F
import json
# 测试集
dataset = load_dataset("json", data_files="/data3/wcr/LTW/c4_subset_500.jsonl")
dataset=dataset["train"]
# dataset=dataset.select(range(1))
print(dataset)
gamma=0.25
delta=3

ckpt_path="/data3/wcr/LTW/ckpt/tmp/selective_network_epoch0_step2000.pth"
def get_first_n_words(text, n=200):
    words = text.split()  
    return ' '.join(words[:n])  

# %%
torch.cuda.set_device(5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path="/data3/wcr/my_project/models/opt-6.7b"


model= AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16).to(device)
tokenizer=AutoTokenizer.from_pretrained(model_path)
model.eval()

# if llama
tokenizer.pad_token = tokenizer.eos_token


# semantic_model= AutoModel.from_pretrained("/data3/wcr/my_lab/hf_models/simcse-roberta-base", torch_dtype=torch.float16).to(device)
# semantic_tokenizer=AutoTokenizer.from_pretrained("/data3/wcr/my_lab/hf_models/simcse-roberta-base")
# semantic_model.eval()

from sentence_transformers import SentenceTransformer, util
import torch


semantic_model = SentenceTransformer('/data3/wcr/my_lab/hf_models/all-mpnet-base-v2')  

# %%
from transformers import OPTForCausalLM, AutoTokenizer,LogitsProcessorList
import torch
import math


if model_path=="/data1/public/models/gpt-j/":
    ppl_model=model
    ppl_tokenizer=tokenizer
else:
    model_name = "/data3/wcr/my_project/selective_watermark_based_on_semantic/models/opt-1.3b"
    ppl_model = OPTForCausalLM.from_pretrained(model_name).to(device)
    ppl_tokenizer = AutoTokenizer.from_pretrained(model_name)

def calculate_perplexity(text):

    inputs = ppl_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():

        outputs = ppl_model(**inputs, labels=inputs["input_ids"])
        

        loss = outputs.loss 
        

    perplexity = torch.exp(loss)  
    return perplexity.item()


def calculate_sim(text1,text2):

    # input_ids=semantic_tokenizer.encode(text1, padding=False, return_tensors='pt',add_special_tokens=False).to(device)
    # sem_embed=semantic_model(input_ids=input_ids, output_hidden_states=True, return_dict=True).pooler_output

    # input_ids2=semantic_tokenizer.encode(text2, padding=False, return_tensors='pt',add_special_tokens=False).to(device)
    # sem_embed2=semantic_model(input_ids=input_ids2, output_hidden_states=True, return_dict=True).pooler_output

    # cos_sim = F.cosine_similarity(sem_embed, sem_embed2, dim=1)
    embedding1 = semantic_model.encode(text1, convert_to_tensor=True)
    embedding2 = semantic_model.encode(text2, convert_to_tensor=True)
    
    cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)
    return cosine_similarity.item()


human_ans=[]
for data in dataset:
    text=data['text']  
    human_ans.append(get_first_n_words(text[300:],200))
    
# %%
from kgw_watermark import WatermarkLogitsProcessor
from sweet import SweetLogitsProcessor
gen_kwargs = {
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 100,
            "min_new_tokens":175,
            "repetition_penalty":1,
            "no_repeat_ngram_size" : 8,
            "max_new_tokens":225

    }
nowm_ans=[]
nowm_ppl=[]
nowm_sim=[]

for data in dataset:
    text=data['text']    
    input_text=text[:300]
    input_ids =tokenizer.encode(input_text, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
    output=model.generate(input_ids=input_ids, attention_mask=attention_mask,pad_token_id=tokenizer.eos_token_id,**gen_kwargs)
    output_text=tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    nowm_ans.append(output_text)
    


count=0
for text in nowm_ans:
    perplexity = calculate_perplexity(text)
    # print(f"Perplexity: {perplexity}")
    nowm_ppl.append(perplexity)
    sim=calculate_sim(human_ans[count],text)
    nowm_sim.append(sim)
    count+=1
    
#print avg perplexity
print(f"Avg Perplexity: {sum(nowm_ppl)/len(nowm_ppl)}")



wm = Watermark(checkpoint_path=ckpt_path,device=device,k=6,model=model,tokenizer=tokenizer, max_new_tokens= 225,min_new_tokens=175,embed_unigram_wm=True)
my_wm_ans=[]
my_wm_ppl=[]
my_wm_sim=[]
for data in dataset:
    text=data['text']    
    input_text=text[:300]
    output=wm.generate_watermark(input_text,gamma,delta)
    # print(output)
    output=output[0]
    my_wm_ans.append(output)

count=0
for text in my_wm_ans:
    perplexity = calculate_perplexity(text)
    # print(f"Perplexity: {perplexity}")
    my_wm_ppl.append(perplexity)

    sim=calculate_sim(human_ans[count],text)
    my_wm_sim.append(sim)
    count+=1

    
#print avg perplexity
print(f"Avg Perplexity: {sum(my_wm_ppl)/len(my_wm_ppl)}")




from kgw_watermark import WatermarkLogitsProcessor
from sweet import SweetLogitsProcessor
gen_kwargs = {
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 100,
            "min_new_tokens":175,
            "repetition_penalty":1,
            "no_repeat_ngram_size" : 8,
            "max_new_tokens":225

    }
kgw_ans=[]
kgw_ppl=[]
kgw_sim=[]
watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                            gamma=gamma,
                                                            delta=delta)
gen_kwargs["logits_processor"] = LogitsProcessorList(
                [watermark_processor]
            )


for data in dataset:
    text=data['text']    
    input_text=text[:300]
    input_ids =tokenizer.encode(input_text, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
    output=model.generate(input_ids=input_ids, attention_mask=attention_mask,pad_token_id=tokenizer.eos_token_id,**gen_kwargs)
    output_text=tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    kgw_ans.append(output_text)

count=0
for text in kgw_ans:
    perplexity = calculate_perplexity(text)
    # print(f"Perplexity: {perplexity}")
    kgw_ppl.append(perplexity)

    sim=calculate_sim(human_ans[count],text)
    kgw_sim.append(sim)
    count+=1
    
#print avg perplexity
print(f"Avg Perplexity: {sum(kgw_ppl)/len(kgw_ppl)}")

# %%
sweet_ans=[]
sweet_ppl=[]
sweet_sim=[]
entropy_threshold=1.2
gen_kwargs = {
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 100,
            "min_new_tokens":175,
            "repetition_penalty":1,
            "no_repeat_ngram_size" : 8,
            "max_new_tokens":225

    }

sweet_processor = SweetLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                            gamma=gamma,
                                                            delta=delta,
                                                            entropy_threshold=entropy_threshold)
gen_kwargs["logits_processor"] = LogitsProcessorList(
                [sweet_processor]
            )



for data in dataset:
    text=data['text']    
    input_text=text[:300]
    input_ids =tokenizer.encode(input_text, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
    output=model.generate(input_ids=input_ids, attention_mask=attention_mask,pad_token_id=tokenizer.eos_token_id,**gen_kwargs)
    output_text=tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    sweet_ans.append(output_text)

count=0
for text in sweet_ans:
    perplexity = calculate_perplexity(text)
    # print(f"Perplexity: {perplexity}")
    sweet_ppl.append(perplexity)
    sim=calculate_sim(human_ans[count],text)
    sweet_sim.append(sim)
    count+=1
    
#print avg perplexity
print(f"Avg Perplexity: {sum(sweet_ppl)/len(sweet_ppl)}")


wm = Watermark(checkpoint_path=ckpt_path,device=device,k=6,model=model,tokenizer=tokenizer, max_new_tokens= 225,min_new_tokens=175)
my_wm_ans1=[]
my_wm_ppl1=[]
my_wm_sim1=[]
for data in dataset:
    text=data['text']    
    input_text=text[:300]
    output=wm.generate_watermark(input_text,gamma,delta)
    # print(output)
    output=output[0]
    my_wm_ans1.append(output)

count=0
for text in my_wm_ans1:
    perplexity = calculate_perplexity(text)
    # print(f"Perplexity: {perplexity}")
    my_wm_ppl1.append(perplexity)

    sim=calculate_sim(human_ans[count],text)
    my_wm_sim1.append(sim)
    count+=1

    
#print avg perplexity
print(f"Avg Perplexity: {sum(my_wm_ppl1)/len(my_wm_ppl1)}")




# %%
# 保存信息
eval_ans={
    "avg_nowm":sum(nowm_ppl)/len(nowm_ppl),
    "avg_nowm_sim":sum(nowm_sim)/len(nowm_sim),
    "avg_ourwm":sum(my_wm_ppl)/len(my_wm_ppl),
    "avg_ourwm_sim":sum(my_wm_sim)/len(my_wm_sim),
    "avg_kgw":sum(kgw_ppl)/len(kgw_ppl),
    "avg_kgw_sim":sum(kgw_sim)/len(kgw_sim),
    "avg_sweet":sum(sweet_ppl)/len(sweet_ppl),
    "avg_sweet_sim":sum(sweet_sim)/len(sweet_sim),
    "avg_ourwm_1":sum(my_wm_ppl1)/len(my_wm_ppl1),
    "avg_ourwm_1_sim":sum(my_wm_sim1)/len(my_wm_sim1),
    "ourwm_ppl":my_wm_ppl,
    "kgw_ppl":kgw_ppl,
    "sweet_ppl":sweet_ppl,
    "nowm_ppl":nowm_ppl,
    "ourwm_sim":my_wm_sim,
    "kgw_sim":kgw_sim,
    "sweet_sim":sweet_sim,
    "nowm_sim":nowm_sim,
    "ourwm_1_ppl":my_wm_ppl1,
    "ourwm_1_sim":my_wm_sim1
}


import json
with open('/data3/wcr/LTW/eval_records/ppl.json', 'w') as f:
    json.dump(eval_ans, f)


