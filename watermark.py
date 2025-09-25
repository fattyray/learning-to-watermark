from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel
from torch.nn import functional as F
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.generation_util import *
from utils.selector_network import SelectorNetwork
from utils.watermark_utils import get_batched_green_mask,get_green_id,get_green_id_unigram,get_batched_green_mask_unigram
from utils.detect_utils import WatermarkDetector



class Watermark():
    def __init__(self,
                 
                device:torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                model_path:str=None,
                semantic_model_path:str="/data3/wcr/my_lab/hf_models/simcse-roberta-base",
                checkpoint_path:str="/data3/wcr/my_project/selective_watermark_based_on_semantic/ckpt/selective_network_init1.pth",
                top_k = 100,
                top_p= 0.95,
                repetition_penalty = 1,
                no_repeat_ngram_size = 8,
                max_new_tokens= 225,
                min_new_tokens= 175,
                k=6,
                model:AutoModelForCausalLM=None,
                tokenizer:AutoTokenizer=None,
                print_ans=False,
                record_entropy=False,
                entropy_record_path=None,
                embed_unigram_wm=False,


                ):
        self.device = device
        self.model_path = model_path
        
        if model is not None:
            self.model = model
            self.tokenizer=tokenizer
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            self.tokenizer= AutoTokenizer.from_pretrained(model_path)
        self.semantic_model = AutoModel.from_pretrained(semantic_model_path).to(device)
        self.semantic_model_tokenizer = AutoTokenizer.from_pretrained(semantic_model_path)
        self.selector_network = SelectorNetwork()
        self.selector_network.load_state_dict(torch.load(checkpoint_path))
        self.selector_network.to(device)
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.k = k
        self.print_ans=print_ans
        self.record_entropy=record_entropy
        self.entropy_record_path=entropy_record_path
        self.embed_unigram_wm=embed_unigram_wm
        hash_key =  15485863
        torch.manual_seed(hash_key)
        torch.cuda.manual_seed(hash_key)


    def generate_unwatermark(self, text):


        # 对文本进行 tokenization 并移到设备

        input_ids = self.tokenizer.encode(text, padding=True, return_tensors='pt').to(self.device)
        attn = torch.ones_like(input_ids)


        output_ids = torch.tensor([[]], dtype=torch.int64, device=self.device)

        past = None

        # 逐步生成新 tokens
        for i in range(self.max_new_tokens):
            with torch.no_grad():
                if past:
                    output = self.model(input_ids=input_ids[:, -1:], attention_mask=attn, past_key_values=past)
                else:
                    output = self.model(input_ids=input_ids, attention_mask=attn)
            
            logits = output.logits[:, -1, :]

            postprocess_next_token_scores(logits, 1, 1, output_ids, repetition_penalty=self.repetition_penalty, no_repeat_ngram_size=self.no_repeat_ngram_size,min_new_tokens=self.min_new_tokens,eos_id=self.tokenizer.eos_token_id)
            logits = top_k_top_p_filtering(logits, top_k=self.top_k, top_p=self.top_p)
            probs = torch.nn.functional.softmax(logits, dim=-1)  
            new_tokens = torch.multinomial(probs, num_samples=1)  # 每次生成一个 token
            input_ids = torch.cat((input_ids, new_tokens), dim=-1)
            output_ids = torch.cat((output_ids, new_tokens), dim=-1)
            attn = torch.cat((attn, attn.new_ones((attn.shape[0], 1))), dim=-1)
            past = output.past_key_values
            if  output_ids[0][-1]== self.tokenizer.eos_token_id:
                break

        # 解码生成的文本
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        if self.print_ans:
            print("Generated Text:")
            print(output_text)
        return output_text


    def generate_watermark(self, text,gamma,delta):


        # 对文本进行 tokenization 并移到设备

        input_ids = self.tokenizer.encode(text, padding=True, return_tensors='pt').to(self.device)
        attn = torch.ones_like(input_ids)
        output_ids = torch.tensor([[]], dtype=torch.int64, device=self.device)
        past = None

        entropy_record=[]   
        watermarked_record=torch.empty((1,0),dtype=torch.float32).to(self.device)
        input_ids_sm=self.semantic_model_tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').to(self.device)
        sm_embedding = self.semantic_model.get_input_embeddings()(input_ids_sm).to(self.device)
        sem_embed=self.semantic_model(inputs_embeds=sm_embedding[:,-self.k:,:], attention_mask=attn[:,-self.k:], output_hidden_states=True, return_dict=True).pooler_output
        # 逐步生成新 tokens
        for i in range(self.max_new_tokens):
            with torch.no_grad():
                if past:
                    output = self.model(input_ids=input_ids[:, -1:], attention_mask=attn, past_key_values=past)
                else:
                    output = self.model(input_ids=input_ids, attention_mask=attn)
            
            logits = output.logits[:, -1, :]
          

            # add watermark

            #########Watermark Info Collector Module#######

            #calculate the   Shannon  entropy of the new logits(following the work of SWEET))
            raw_probs = F.softmax(logits, dim=-1)
            entropy = -torch.where(raw_probs > 0, raw_probs * raw_probs.log(), raw_probs.new([0.0])).sum(dim=-1)
            entropy = entropy.unsqueeze(dim=-1).to(self.device)

            # calculate the watermarked ratio
            if watermarked_record.shape[-1] == 0:
                watered_ratio = torch.full((watermarked_record.shape[0], 1), float(0)).to(self.device)
            else:
                watered_ratio = watermarked_record.float().mean(dim=-1, keepdim=True).to(self.device) 

            # using the semantic embedding info to deside whether to add the watermark or not
            network_input=torch.cat((entropy, watered_ratio, sem_embed), dim=-1)

            #########Selecor Network#######

            with torch.no_grad():
                out=self.selector_network(network_input)

            #########Adaptive threshold#######

            if watered_ratio.mean().item()<0.25:
                threshold=0.4
            elif watered_ratio.mean().item()>0.6:
                threshold=0.65
            else:
                threshold=0.5

            entropy_record.append( (entropy.mean().item(), out.mean().item()))


            watermark_mask=torch.where(out>threshold, torch.tensor(1), torch.tensor(0)).to(self.device)
            watermarked_record = torch.cat((watermarked_record, watermark_mask), dim=-1)

            
            if self.embed_unigram_wm:
                green_mask=get_batched_green_mask_unigram(logits,gamma)
            else:
                green_mask=get_batched_green_mask(logits,gamma, input_ids[:, -1])
            #combine those two together to form a selective watermark's mask
            watermark_mask.expand(-1,green_mask.shape[-1])
            result_mask=watermark_mask*green_mask 
            logits=logits+delta*result_mask


            postprocess_next_token_scores(logits, 1, 1, output_ids, repetition_penalty=self.repetition_penalty, no_repeat_ngram_size=self.no_repeat_ngram_size,min_new_tokens=self.min_new_tokens,eos_id=self.tokenizer.eos_token_id)
            logits = top_k_top_p_filtering(logits, top_k=self.top_k, top_p=self.top_p)
            probs = torch.nn.functional.softmax(logits, dim=-1)  
            new_id = torch.multinomial(probs, num_samples=1)  # 每次生成一个 token
            input_ids = torch.cat((input_ids, new_id), dim=-1)
            output_ids = torch.cat((output_ids, new_id), dim=-1)
            attn = torch.cat((attn, attn.new_ones((attn.shape[0], 1))), dim=-1)

            new_token=self.tokenizer.decode(new_id.tolist()[0])
            sm_new_id=self.semantic_model_tokenizer.encode(new_token, padding=False, return_tensors='pt',add_special_tokens=False).to(self.device)
            if sm_new_id.numel()>0:
                input_ids_sm=torch.cat((input_ids_sm, sm_new_id), dim=-1)
                sm_embedding=torch.cat((sm_embedding, self.semantic_model.get_input_embeddings()(sm_new_id).to(self.device)), dim=1)
            sem_embed=self.semantic_model(inputs_embeds=sm_embedding[:,-self.k:,:], attention_mask=attn[:,-self.k:], output_hidden_states=True, return_dict=True).pooler_output
            past = output.past_key_values
            if  output_ids[0][-1]== self.tokenizer.eos_token_id:
                break
        if self.record_entropy and self.entropy_record_path is not None:
                # create json file if not exist,else load it
                if not os.path.exists(self.entropy_record_path):
                    with open(self.entropy_record_path, 'w') as f:
                        json.dump([], f)
                with open(self.entropy_record_path, 'r') as f:
                    entropy_record_ = json.load(f)
                entropy_record_+=entropy_record
                with open(self.entropy_record_path, 'w') as f:
                    json.dump(entropy_record_, f)
   
        # 解码生成的文本
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        if self.print_ans:
            print("Generated Text:")
            print(output_text)
        return output_text
                
class Detector(WatermarkDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def _score_sequence(
        self,
        input_ids: torch.Tensor,
        prefix_len: int,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_watermarking_fraction: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_p_value: bool = True,
    ):
        score_dict = dict()
        if self.ignore_repeated_bigrams:
            raise NotImplementedError("not implemented")
        prefix_len = max(self.min_prefix_len, prefix_len)

        num_tokens_generated = len(input_ids) - prefix_len
        if num_tokens_generated < 1:
            print(f"no generated tokens, skipping")
            score_dict["invalid"] = True
            return score_dict
        entropy=self.calculate_entropy(self.model,input_ids)

        nums_tokens_scored = 0 
        nums_green_tokens = 0
        green_token_mask=[]
        watermarked_record=torch.empty((1,0),dtype=torch.float32).to(self.device)
        watered_ratio = torch.full((watermarked_record.shape[0], 1), float(0)).to(self.device)      
        input_ids_sm=self.semantic_model_tokenizer.encode(self.tokenizer.decode(input_ids[max(0,prefix_len-self.k):prefix_len]), padding=True, return_tensors='pt',add_special_tokens=False).to(self.device)
        sm_embedding = self.semantic_model.get_input_embeddings()(input_ids_sm).to(self.device)

        attn=torch.ones(1,input_ids.shape[0])

        for i in range(num_tokens_generated):


            id_pos=prefix_len+i
            sem_embed=self.semantic_model(inputs_embeds=sm_embedding[:,-self.k:,:], attention_mask=attn[:,max(0,id_pos-self.k):id_pos], output_hidden_states=True, return_dict=True).pooler_output
            ent=entropy[id_pos-1]
            ent=ent.unsqueeze(0).unsqueeze(1).to(self.device)
            network_input=torch.cat((ent, watered_ratio, sem_embed), dim=-1)
            with torch.no_grad():
                out=self.selector_network(network_input)
            if watered_ratio.mean().item()<0.25:
                threshold=0.4
            elif watered_ratio.mean().item()>0.6:
                threshold=0.65
            else:
                threshold=0.5
            watermark_mask=torch.where(out>threshold, torch.tensor(1), torch.tensor(0)).to(self.device)
            watermarked_record = torch.cat((watermarked_record, watermark_mask), dim=-1)
            watered_ratio = watermarked_record.float().mean(dim=-1, keepdim=True).to(self.device)
            sm_new_id=self.semantic_model_tokenizer.encode(self.tokenizer.decode(input_ids[id_pos]), padding=False, return_tensors='pt',add_special_tokens=False).to(self.device)
            if sm_new_id.numel()>0:
                sm_embedding=torch.cat((sm_embedding, self.semantic_model.get_input_embeddings()(sm_new_id).to(self.device)), dim=1)
            if watermark_mask.sum() > 0:          
                nums_tokens_scored += 1
                if self.embed_unigram_wm :
                    greenlist_ids=get_green_id_unigram(self.tokenizer.vocab_size,self.device,self.gamma)
                else:
                    greenlist_ids = get_green_id( self.tokenizer.vocab_size,self.device,self.gamma,input_ids[id_pos-1].item())
                if input_ids[id_pos].cpu() in greenlist_ids:
                    nums_green_tokens +=1
                    green_token_mask.append(True)
                else:
                    green_token_mask.append(False)
            else:
                green_token_mask.append(False)


        if nums_tokens_scored < 1:
            assert nums_tokens_scored == 0
            # regarding as human generated
            return {
                "num_tokens_generated": num_tokens_generated,
                "num_tokens_scored": 0,
                "num_green_tokens": 0,
                "watermarking_fraction": 0,
                "green_fraction": 0,
                "z_score": -100.0,
                "p_value": 1,
            }
        
        score_dict.update(dict(num_tokens_generated=num_tokens_generated))
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=nums_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=nums_green_tokens))
        if return_watermarking_fraction:
            score_dict.update(
                dict(watermarking_fraction=(nums_tokens_scored / num_tokens_generated))
            )
        if return_green_fraction:
            score_dict.update(
                dict(green_fraction=(nums_green_tokens / nums_tokens_scored))
            )
        if return_z_score:
            score_dict.update(
                dict(
                    z_score=self._compute_z_score(nums_green_tokens, nums_tokens_scored)
                )
            )
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(nums_green_tokens, nums_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask))

        return score_dict

                










