import json
from dipper import DipperParaphraser

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def collect_data(list_output_dir, dict_output_dir, dict_key, names):
    """从 JSON 文件中提取数据并按顺序存入字典"""
    result = {}
    
    # 处理 list_output_dir
    for idx, file_path in enumerate(list_output_dir):
        data = load_json(file_path)
        result[names[idx]] = data if isinstance(data, list) else []  # 确保数据是列表
    
    # 处理 dict_output_dir
    for idx, file_path in enumerate(dict_output_dir):
        data = load_json(file_path)
        key = dict_key[idx]
        name_index = len(list_output_dir) + idx  # names 的索引
        
        if key in data and isinstance(data[key], list):
            result[names[name_index]] = data[key]
        else:
            result[names[name_index]] = []  # 处理缺失的 key 或非列表情况
    
    return result

# 定义文件路径和键
# list_output_dir = [
#     "/data3/wcr/my_project/selective_watermark_based_on_semantic/eval_records/our_wm_output_embed_kgw1.json",
#     "/data3/wcr/my_project/selective_watermark_based_on_semantic/eval_records/our_wm_output_embed_kgw0.json",
#     "/data3/wcr/my_project/selective_watermark_based_on_semantic/eval_records/sweet_wm_ouput.json",
#     "/data3/wcr/my_project/selective_watermark_based_on_semantic/eval_records/kgw_wm_output.json"
# ]
# dict_output_dir = [
#     "/data3/wcr/others work/Unigram-Watermark-main/unigram_output.json",
#     "/data3/wcr/others work/watermark-main/output.json",
#     "/data3/wcr/others work/TS_watermark-main/ts_output.json"
# ]

list_output_dir = [
    "/data3/wcr/my_project/selective_watermark_based_on_semantic/eval_records/our_wm_output_embed_kgw1_gpt-j.json",
    "/data3/wcr/my_project/selective_watermark_based_on_semantic/eval_records/our_wm_output_embed_kgw0_gpt-j.json",
    "/data3/wcr/my_project/selective_watermark_based_on_semantic/eval_records/sweet_wm_ouput_gpt-j.json",
    "/data3/wcr/my_project/selective_watermark_based_on_semantic/eval_records/kgw_wm_output_gpt-j.json"
]
dict_output_dir = [
    "/data3/wcr/others work/Unigram-Watermark-main/unigram_output_gpt-j.json",
    "/data3/wcr/others work/watermark-main/output_gpt-j.json",
    "/data3/wcr/others work/TS_watermark-main/ts_output_gpt-j.json"
]
dict_key = ["output_noprompt", "output_noprompt", "output_noprompt"]
names = ["our_wm_kgw1", "our_wm_kgw0","sweet", "kgw", "unigram", "exp-edit",  "TS_watermark"]

# 获取数据
data_dict = collect_data(list_output_dir, dict_output_dir, dict_key, names)

# 打印或使用数据
output_path = "/data3/wcr/my_project/selective_watermark_based_on_semantic/eval/wms_output_gpt-j.json"
try:
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)
        print(f"保存 成功")   
except Exception as e:
    print(f"保存 JSON 失败: {e}")


dipper_output={}
for key in data_dict:
    dipper_output[key]=[]   

print(f"start dipper attack")   
# 使用dipper进行paraphrase attack
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
hash_key = 15485863
torch.manual_seed(hash_key)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dp = DipperParaphraser()

from datasets import load_dataset, Dataset
dataset = load_dataset("json", data_files="/data3/wcr/my_project/selective_watermark_based_on_semantic/c4_subset_500.jsonl")
dataset=dataset["train"]
count=0
for data in dataset:
    text=data['text']    
    prompt=text[:300]
    for key in data_dict:   
            output_l60_sample = dp.paraphrase(data_dict[key][count], lex_diversity=60, order_diversity=0, prefix=prompt, do_sample=True, top_p=0.75, top_k=None, max_length=300)
            dipper_output[key].append(output_l60_sample)
    count+=1
# 保存结果
output_para_path = "/data3/wcr/my_project/selective_watermark_based_on_semantic/eval/dipper_attack_output_gpt-j.json"
try:
    with open(output_para_path, 'w', encoding='utf-8') as f:
        json.dump(dipper_output, f, ensure_ascii=False, indent=4)
        print(f"保存 成功")   
except Exception as e:
    print(f"保存 JSON 失败: {e}")   
    
