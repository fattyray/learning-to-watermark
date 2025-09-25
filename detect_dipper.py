import torch
from transformers import AutoTokenizer
from watermark import Detector
from sweet import SweetLogitsProcessor,SweetDetector
from kgw_watermark import WatermarkLogitsProcessor,WatermarkDetector
from transformers import AutoTokenizer,AutoModelForCausalLM,LogitsProcessorList
import torch
import numpy as np

torch.cuda.set_device(4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path="/data1/public/models/gpt-j"
model= AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer=AutoTokenizer.from_pretrained(model_path)
model.eval()

gamma=0.25
delta=3
z_threshold=4
my_wm_z=[]
my_wm_z2=[]
sweet_wm_z=[]
kgw_wm_z=[]

watermark_detector = Detector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=gamma,
                                        tokenizer=tokenizer,
                                        z_threshold=z_threshold,
                                        model=model,
                                        k=6,
                                        checkpoint_path="/data3/wcr/my_project/selective_watermark_based_on_semantic/ckpt/tmp/selective_network_epoch0_step2000.pth"             
                                        )

watermark_detector2 = Detector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=gamma,
                                        tokenizer=tokenizer,
                                        z_threshold=z_threshold,
                                        model=model,
                                        k=6,
                                        checkpoint_path="/data3/wcr/my_project/selective_watermark_based_on_semantic/ckpt/tmp/selective_network_epoch0_step2000.pth",
                                        embed_unigram_wm=True,             
                                        )

entropy_threshold=1.2 # entropy_threshold follows the paper of sweet
sweet_detector = SweetDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=gamma,
                                        tokenizer=tokenizer,
                                        z_threshold=z_threshold,
                                        entropy_threshold=entropy_threshold)

def calculate_entropy(model, tokenized_text):
    with torch.no_grad():

        output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
        probs = torch.softmax(output.logits, dim=-1)
        entropy = -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(dim=-1)
        return entropy[0].cpu().tolist()

kgw_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=gamma,
                                        tokenizer=tokenizer,
                                        z_threshold=z_threshold)
# load json file
import json
with open('/data3/wcr/my_project/selective_watermark_based_on_semantic/eval/dipper_attack_output_gpt-j.json', 'r') as f:
    dipper_output = json.load(f)



from datasets import load_dataset, Dataset
dataset = load_dataset("json", data_files="/data3/wcr/my_project/selective_watermark_based_on_semantic/c4_subset_500.jsonl")
dataset=dataset["train"]
count=0
for data in dataset:
    text=data['text']    
    input_text=text[:300]
    #eval our_wm after dipper attack
    output=dipper_output["our_wm_kgw1"][count]
    output=input_text+output
    tokenized_input=tokenizer.encode(input_text,  return_tensors='pt',add_special_tokens=False).to(device)
    tokenized_output=tokenizer.encode(output, return_tensors='pt').to(device)
    tokenized_output=tokenized_output[0]
    tokenized_input=tokenized_input[0]
    detection_result=watermark_detector.detect(tokenized_output,tokenized_input)
    my_wm_z.append(detection_result['z_score'])

    # eval the embedding unigram version of our_wm after dipper attack
    output=dipper_output["our_wm_kgw0"][count]
    output=input_text+output
    tokenized_input=tokenizer.encode(input_text,  return_tensors='pt',add_special_tokens=False).to(device)
    tokenized_output=tokenizer.encode(output, return_tensors='pt').to(device)
    tokenized_output=tokenized_output[0]
    tokenized_input=tokenized_input[0]
    detection_result=watermark_detector2.detect(tokenized_output,tokenized_input)
    my_wm_z2.append(detection_result['z_score'])


    #eval sweet_wm after dipper attack
    output=dipper_output["sweet"][count]
    output=input_text+output
    tokenized_output=tokenizer.encode(output, return_tensors='pt').to(device)
    tokenized_output=tokenized_output[0]
    entropy = calculate_entropy(model, tokenized_output)
                        # we need to shift entropy to the right, so the first item is dummy
    entropy = [0] + entropy[:-1]
    detection_result=sweet_detector.detect(tokenized_output,tokenized_input,entropy=entropy)
    sweet_wm_z.append(detection_result['z_score'])

    #eval kgw_wm after dipper attack
    output=dipper_output["kgw"][count]
    output=input_text+output
    tokenized_output=tokenizer.encode(output, return_tensors='pt').to(device)
    tokenized_output=tokenized_output[0]

    detection_result=kgw_detector.detect(tokenized_output,tokenized_input)
    kgw_wm_z.append(detection_result['z_score'])
    count+=1


ans={
    "our_wm_kgw1":my_wm_z,
    "sweet_wm_z":sweet_wm_z,
    "kgw_wm_z":kgw_wm_z,
    "our_wm_kgw0":my_wm_z2
}
# save the result to json file
import json
with open('/data3/wcr/my_project/selective_watermark_based_on_semantic/eval_records/dipper_score_gpt-j.json', 'w') as f:
    json.dump(ans, f)
