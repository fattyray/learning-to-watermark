from datasets import load_dataset, Dataset
from watermark import Watermark
import torch
from transformers import AutoTokenizer
from watermark import Detector
from sweet import SweetLogitsProcessor,SweetDetector
from kgw_watermark import WatermarkLogitsProcessor,WatermarkDetector
from transformers import AutoTokenizer,AutoModelForCausalLM,LogitsProcessorList
import torch
import numpy as np
dataset = load_dataset("json", data_files="/data3/wcr/LTW/c4_subset_500.jsonl")
dataset=dataset["train"]
# dataset=dataset.select(range(1))
print(dataset)
gamma=0.25
delta=3

my_wm_z=[]
sweet_wm_z=[]
kgw_wm_z=[]

un_wm_z=[]
um_sweet_z=[]
um_kgw_z=[]
un_wm_output=[]
our_wm_output=[]
kgw_wm_output=[]
sweet_wm_ouput=[]
def calculate_entropy(model, tokenized_text):
    with torch.no_grad():

        output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
        probs = torch.softmax(output.logits, dim=-1)
        entropy = -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(dim=-1)
        return entropy[0].cpu().tolist()

torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path="/data3/wcr/my_project/selective_watermark_based_on_semantic/models/opt-6.7b"


model= AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16).to(device)
tokenizer=AutoTokenizer.from_pretrained(model_path)
model.eval()
# if llama
tokenizer.pad_token = tokenizer.eos_token

wm = Watermark(checkpoint_path="/data3/wcr/LTW/ckpt/tmp/selective_network_epoch0_step2000.pth",device=device,k=6,model=model,tokenizer=tokenizer, max_new_tokens= 225,min_new_tokens=175,embed_unigram_wm=True,   )
watermark_detector = Detector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=gamma,
                                        tokenizer=tokenizer,
                                        z_threshold=4,
                                        model=model,
                                        k=6,
                                        checkpoint_path="/data3/wcr/my_project/selective_watermark_based_on_semantic/ckpt/tmp/selective_network_epoch0_step2000.pth",
                                        embed_unigram_wm=True,   
                                    
                                        )
for data in dataset:
    text=data['text']    
    input_text=text[:300]
    output=wm.generate_watermark(input_text,gamma,delta)
    our_wm_output.append(output[0])
    # print(output)
    output=input_text+output[0]
    tokenized_input=tokenizer.encode(input_text,  return_tensors='pt',add_special_tokens=False).to(device)
    tokenized_output=tokenizer.encode(output, return_tensors='pt').to(device)
    tokenized_output=tokenized_output[0]
    tokenized_input=tokenized_input[0]
    detection_result=watermark_detector.detect(tokenized_output,tokenized_input)
    my_wm_z.append(detection_result['z_score'])


    output=wm.generate_unwatermark(input_text)
    un_wm_output.append(output[0])
    output=input_text+output[0]
    tokenized_input=tokenizer.encode(input_text,  return_tensors='pt',add_special_tokens=False).to(device)
    tokenized_output=tokenizer.encode(output, return_tensors='pt').to(device)
    tokenized_output=tokenized_output[0]
    tokenized_input=tokenized_input[0]
    detection_result=watermark_detector.detect(tokenized_output,tokenized_input)
    un_wm_z.append(detection_result['z_score'])


gen_kwargs = {
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 100,
            "min_new_tokens":175,
            "repetition_penalty":1,
            "no_repeat_ngram_size" : 8,
            "max_new_tokens":225

    }

# eval kgw
z_threshold=4
watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                            gamma=gamma,
                                                            delta=delta)
gen_kwargs["logits_processor"] = LogitsProcessorList(
                [watermark_processor]
            )
        
watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=gamma,
                                        tokenizer=tokenizer,
                                        z_threshold=z_threshold)
count=0
for data in dataset:
    text=data['text']    
    input_text=text[:300]
    input_ids =tokenizer.encode(input_text, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
    output=model.generate(input_ids=input_ids, attention_mask=attention_mask,pad_token_id=tokenizer.eos_token_id,**gen_kwargs)
    output_text=tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    kgw_wm_output.append(output_text)
    output=output[0]
    detection_result=watermark_detector.detect(output,input_ids[0])
    kgw_wm_z.append(detection_result['z_score'])
    um_text=input_text+un_wm_output[count]
    output_ids =tokenizer.encode(um_text, return_tensors='pt').to(device)
    detection_result=watermark_detector.detect(output_ids[0],input_ids[0])
    count+=1
    um_kgw_z.append(detection_result['z_score'])


entropy_threshold=1.2 # entropy_threshold follows the paper of sweet
sweet_processor = SweetLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                            gamma=gamma,
                                                            delta=delta,
                                                            entropy_threshold=entropy_threshold)
gen_kwargs["logits_processor"] = LogitsProcessorList(
                [sweet_processor]
            )
        
watermark_detector = SweetDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=gamma,
                                        tokenizer=tokenizer,
                                        z_threshold=z_threshold,
                                        entropy_threshold=entropy_threshold)


count=0
for data in dataset:
    text=data['text']    
    input_text=text[:300]
    input_ids =tokenizer.encode(input_text, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
    output=model.generate(input_ids=input_ids, attention_mask=attention_mask,pad_token_id=tokenizer.eos_token_id,**gen_kwargs)
    output_text=tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    sweet_wm_ouput.append(output_text)
    output=output[0]
    entropy = calculate_entropy(model, output)
                        # we need to shift entropy to the right, so the first item is dummy
    entropy = [0] + entropy[:-1]
    detection_result=watermark_detector.detect(output,input_ids[0],entropy=entropy)
    sweet_wm_z.append(detection_result['z_score'])
    um_text=input_text+un_wm_output[count]
    output_ids =tokenizer.encode(um_text, return_tensors='pt').to(device)
    entropy = calculate_entropy(model, output_ids[0])
                        # we need to shift entropy to the right, so the first item is dummy
    entropy = [0] + entropy[:-1]
    detection_result=watermark_detector.detect(output_ids[0],input_ids[0],entropy=entropy)
    count+=1
    um_sweet_z.append(detection_result['z_score'])




sorted_un_wm_z = np.sort(un_wm_z)[::-1]


threshold_0_fpr = sorted_un_wm_z[int(0 * len(sorted_un_wm_z))]
threshold_1_fpr = sorted_un_wm_z[int(0.01 * len(sorted_un_wm_z))]
threshold_2_fpr = sorted_un_wm_z[int(0.02 * len(sorted_un_wm_z))]


def calculate_tpr(threshold, my_wm_z):

    tp = np.sum(np.array(my_wm_z) > threshold) 
    tpr = 1.0*tp / len(my_wm_z)  # TPR = TP / (TP + FN)
    return tpr


tpr_0_fpr = calculate_tpr(threshold_0_fpr, my_wm_z)
tpr_1_fpr = calculate_tpr(threshold_1_fpr, my_wm_z)
tpr_2_fpr = calculate_tpr(threshold_2_fpr, my_wm_z)


eval_ans={
    "tpr@0":tpr_0_fpr,
    "tpr@1":tpr_1_fpr,
    "tpr@2":tpr_2_fpr,
    "my_wm_z":my_wm_z,
    "un_wm_z":un_wm_z

    
}


import json
with open('/data3/wcr/LTW/eval_records/our_tpr_embed_kgw0.json', 'w') as f:
    json.dump(eval_ans, f)

with open('/data3/wcr/LTW/eval_records/un_wm_output.json', 'w') as f:
    json.dump(un_wm_output, f)

with open('/data3/wcr/LTW/eval_records/our_wm_output_embed_kgw0.json', 'w') as f:
    json.dump(our_wm_output, f)

with open('/data3/wcr/LTW/eval_records/kgw_wm_output.json', 'w') as f:
    json.dump(kgw_wm_output, f)

with open('/data3/wcr/LTW/eval_records/sweet_wm_ouput.json', 'w') as f:
    json.dump(sweet_wm_ouput, f)



sorted_un_wm_z = np.sort(um_kgw_z)[::-1]


threshold_0_fpr = sorted_un_wm_z[int(0 * len(sorted_un_wm_z))]
threshold_1_fpr = sorted_un_wm_z[int(0.01 * len(sorted_un_wm_z))]
threshold_2_fpr = sorted_un_wm_z[int(0.02 * len(sorted_un_wm_z))]


tpr_0_fpr = calculate_tpr(threshold_0_fpr, kgw_wm_z)
tpr_1_fpr = calculate_tpr(threshold_1_fpr, kgw_wm_z)
tpr_2_fpr = calculate_tpr(threshold_2_fpr, kgw_wm_z)


eval_ans={
    "tpr@0":tpr_0_fpr,
    "tpr@1":tpr_1_fpr,
    "tpr@2":tpr_2_fpr,
    "kgw_wm_z":kgw_wm_z,
    "un_kgw_z":um_kgw_z

    
}


import json
with open('/data3/wcr/LTW/eval_records/kgw_tpr.json', 'w') as f:
    json.dump(eval_ans, f)




sorted_un_wm_z = np.sort(um_sweet_z)[::-1]


threshold_0_fpr = sorted_un_wm_z[int(0 * len(sorted_un_wm_z))]
threshold_1_fpr = sorted_un_wm_z[int(0.01 * len(sorted_un_wm_z))]
threshold_2_fpr = sorted_un_wm_z[int(0.02 * len(sorted_un_wm_z))]


tpr_0_fpr = calculate_tpr(threshold_0_fpr, sweet_wm_z)
tpr_1_fpr = calculate_tpr(threshold_1_fpr, sweet_wm_z)
tpr_2_fpr = calculate_tpr(threshold_2_fpr, sweet_wm_z)


eval_ans={
    "tpr@0":tpr_0_fpr,
    "tpr@1":tpr_1_fpr,
    "tpr@2":tpr_2_fpr,
    "sweet_wm_z":sweet_wm_z,
    "um_sweet_z":um_sweet_z

    
}

#把结果保存到json文件中
import json
with open('/data3/wcr/LTW/eval_records/sweet_tpr.json', 'w') as f:
    json.dump(eval_ans, f)



my_wm_z=[]
un_wm_z=[]
our_wm_output=[]
wm = Watermark(checkpoint_path="/data3/wcr/LTW/ckpt/tmp/selective_network_epoch0_step2000.pth",device=device,k=6,model=model,tokenizer=tokenizer, max_new_tokens= 225,min_new_tokens=175)
watermark_detector = Detector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=gamma,
                                        tokenizer=tokenizer,
                                        z_threshold=4,
                                        model=model,
                                        k=6,
                                        checkpoint_path="/data3/wcr/LTW/ckpt/tmp/selective_network_epoch0_step2000.pth",
                                         
                                        )
count=0
for data in dataset:
    text=data['text']    
    input_text=text[:300]
    output=wm.generate_watermark(input_text,gamma,delta)
    our_wm_output.append(output[0])
    # print(output)
    output=input_text+output[0]
    tokenized_input=tokenizer.encode(input_text,  return_tensors='pt',add_special_tokens=False).to(device)
    tokenized_output=tokenizer.encode(output, return_tensors='pt').to(device)
    tokenized_output=tokenized_output[0]
    tokenized_input=tokenized_input[0]
    detection_result=watermark_detector.detect(tokenized_output,tokenized_input)
    my_wm_z.append(detection_result['z_score'])

    output=input_text+un_wm_output[count]
    tokenized_input=tokenizer.encode(input_text,  return_tensors='pt',add_special_tokens=False).to(device)
    tokenized_output=tokenizer.encode(output, return_tensors='pt').to(device)
    tokenized_output=tokenized_output[0]
    tokenized_input=tokenized_input[0]
    detection_result=watermark_detector.detect(tokenized_output,tokenized_input)
    un_wm_z.append(detection_result['z_score'])
    count+=1





sorted_un_wm_z = np.sort(un_wm_z)[::-1]


threshold_0_fpr = sorted_un_wm_z[int(0 * len(sorted_un_wm_z))]
threshold_1_fpr = sorted_un_wm_z[int(0.01 * len(sorted_un_wm_z))]
threshold_2_fpr = sorted_un_wm_z[int(0.02 * len(sorted_un_wm_z))]


def calculate_tpr(threshold, my_wm_z):

    tp = np.sum(np.array(my_wm_z) > threshold) 
    tpr = 1.0*tp / len(my_wm_z)  # TPR = TP / (TP + FN)
    return tpr


tpr_0_fpr = calculate_tpr(threshold_0_fpr, my_wm_z)
tpr_1_fpr = calculate_tpr(threshold_1_fpr, my_wm_z)
tpr_2_fpr = calculate_tpr(threshold_2_fpr, my_wm_z)


eval_ans={
    "tpr@0":tpr_0_fpr,
    "tpr@1":tpr_1_fpr,
    "tpr@2":tpr_2_fpr,
    "my_wm_z":my_wm_z,
    "un_wm_z":un_wm_z

    
}


import json
with open('/data3/wcr/LTW/eval_records/our_tpr_embed_kgw1.json', 'w') as f:
    json.dump(eval_ans, f)

with open('/data3/wcr/LTW/eval_records/our_wm_output_embed_kgw1.json', 'w') as f:
    json.dump(our_wm_output, f)
