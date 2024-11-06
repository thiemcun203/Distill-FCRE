import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
from transformers import RobertaModel
from transformers import BertForMaskedLM
class EncodingModel(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        if config.model == 'bert':
            self.encoder = BertModel.from_pretrained(config.bert_path, torch_dtype=torch.bfloat16 if config.dtype == 'bfloat16' else torch.float32).to(config.device)
            self.lm_head = BertForMaskedLM.from_pretrained(config.bert_path, torch_dtype=torch.bfloat16 if config.dtype == 'bfloat16' else torch.float32).to(config.device).cls
        elif config.model == 'roberta':
            self.encoder = RobertaModel.from_pretrained(config.roberta_path, torch_dtype=torch.bfloat16 if config.dtype == 'bfloat16' else torch.float32).to(config.device)
            self.encoder.resize_token_embeddings(config.vocab_size)
        if config.tune == 'prompt':
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.bert_word_embedding = self.encoder.get_input_embeddings()
        self.embedding_dim = self.bert_word_embedding.embedding_dim
        self.prompt_lens = config.prompt_len * config.prompt_num
        self.softprompt_encoder = nn.Embedding(self.prompt_lens, self.embedding_dim).to(config.device)
        # initialize prompt embedding
        self._init_prompt()
        self.prompt_ids = torch.LongTensor(list(range(self.prompt_lens))).to(self.config.device)

        self.info_nce_fc = nn.Linear(self.embedding_dim , self.embedding_dim).to(config.device)


    def infoNCE_f(self, V, C):
        """
        V : B x vocab_size
        C : B x embedding_dim
        """
        try:
            out = self.info_nce_fc(V) # B x embedding_dim
            out = torch.matmul(out , C.t()) # B x B

        except:
            print("V.shape: ", V.shape)
            print("C.shape: ", C.shape)
            print("info_nce_fc: ", self.info_nce_fc)
        return out
    def _init_prompt(self):
        # is is is [e1] is is is [MASK] is is is [e2] is is is
        if self.config.prompt_init == 1:
            prompt_embedding = torch.zeros_like(self.softprompt_encoder.weight).to(self.config.device)
            token_embedding = self.bert_word_embedding.weight[2003]
            prompt_embedding[list(range(self.prompt_lens)), :] = token_embedding.clone().detach()
            for param in self.softprompt_encoder.parameters():
                param.data = prompt_embedding # param.data
       
        # ! @ # [e1] he is as [MASK] * & % [e2] just do it  
        elif self.config.prompt_init == 2:
            prompt_embedding = torch.zeros_like(self.softprompt_encoder.weight).to(self.config.device)
            ids = [999, 1030, 1001, 2002, 2003, 2004, 1008, 1004, 1003, 2074, 2079,  2009]
            for i in range(self.prompt_lens):
                token_embedding = self.bert_word_embedding.weight[ids[i]]
                prompt_embedding[i, :] = token_embedding.clone().detach()
            for param in self.softprompt_encoder.parameters():
                param.data = prompt_embedding # param.data


    def embedding_input(self, input_ids): # (b, max_len)
        input_embedding = self.bert_word_embedding(input_ids) # (b, max_len, h)
        prompt_embedding = self.softprompt_encoder(self.prompt_ids) # (prompt_len, h)

        for i in range(input_ids.size()[0]):
            p = 0
            for j in range(input_ids.size()[1]):
                if input_ids[i][j] == self.config.prompt_token_ids:
                    input_embedding[i][j] = prompt_embedding[p]
                    p += 1

        return input_embedding


    def forward(self, inputs, is_des=False): # (b, max_length)
        batch_size = inputs['ids'].size()[0]
        tensor_range = torch.arange(batch_size) # (b)     
        pattern = self.config.pattern
        if pattern == 'softprompt' or pattern == 'hybridprompt':
            input_embedding = self.embedding_input(inputs['ids'])
            outputs_words = self.encoder(inputs_embeds=input_embedding, attention_mask=inputs['mask'])[0]
        else:
            outputs_words = self.encoder(inputs['ids'], attention_mask=inputs['mask'])[0] # (b, max_length, h)
            # outputs_words_des = self.encoder(inputs['ids_des'], attention_mask=inputs['mask_des'])[0] # (b, max_length, h)


        # return [CLS] hidden
        if pattern == 'cls' or pattern == 'softprompt':
            clss = torch.zeros(batch_size, dtype=torch.long)
            return outputs_words[tensor_range ,clss] # (b, h)

        # return [MASK] hidden
        elif pattern == 'hardprompt' or pattern == 'hybridprompt':
            masks = []
            for i in range(batch_size):
                ids = inputs['ids'][i].cpu().numpy()
                try:
                    mask = np.argwhere(ids == self.config.mask_token_ids)[0][0]
                except:
                    mask = 0
                
                masks.append(mask)
            if is_des:
                average_outputs_words = torch.mean(outputs_words, dim=1)
                return average_outputs_words
            else:
                mask_hidden = outputs_words[tensor_range, torch.tensor(masks)] # (b, h)
                return mask_hidden
            # lm_head_output = self.lm_head(mask_hidden) # (b, max_length, vocab_size)
            # return mask_hidden , average_outputs_words

        # return e1:e2 hidden
        elif pattern == 'marker':
            h1, t1 = [], []
            for i in range(batch_size):
                ids = inputs['ids'][i].cpu().numpy()
                h1_index, t1_index = np.argwhere(ids == self.config.h_ids), np.argwhere(ids == self.config.t_ids)
                h1.append(0) if h1_index.size == 0 else h1.append(h1_index[0][0])
                t1.append(0) if t1_index.size == 0 else t1.append(t1_index[0][0])

            h_state = outputs_words[tensor_range, torch.tensor(h1)] # (b, h)
            t_state = outputs_words[tensor_range, torch.tensor(t1)]

            concerate_h_t = (h_state + t_state) / 2 # (b, h)
            return concerate_h_t

import torch
import torch.nn as nn
from llm2vec import LLM2Vec
from torch import Tensor, device
from llm2vec import LLM2Vec
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from typing import List, Optional

def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

class EncodingModel_LLM2Vec(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            )
        self.dtype = torch.bfloat16 if config.dtype == 'bfloat16' else torch.float32
        self.encoder = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=self.dtype,
            merge_peft=True,
            pooling_mode="mean",
            max_length=256,
            token = "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",
        )
        # print(123)
        # if config.train_llm2vec == True:
        self.encoder.model = self.initialize_peft(
            self.encoder.model,
        )
        # self.vector_linear = nn.Sequential(
        #     nn.Linear(in_features=4096, out_features=768),
        #     nn.Tanh()
        # ).to('cuda', dtype=self.dtype)
            
    def initialize_peft(
        self,
        model,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_modules: Optional[List[str]] = None,
    ):
        if lora_modules is None and model.config.__class__.__name__ in [
            "LlamaConfig",
            "MistralConfig",
            "GemmaConfig",
            "Qwen2Config",
        ]:
            lora_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        elif lora_modules is None:
            raise ValueError("lora_modules must be specified for this model.")

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None,
        )

        model = get_peft_model(model, config)
        print(f"Model's Lora trainable parameters:")
        model.print_trainable_parameters()
        return model

    def forward(self, inputs, is_des = False): # (b, max_length)
        # features = self.encoder.tokenize(inputs['input'])
        features = self.encoder.tokenize(inputs)
        features = batch_to_device(features, self.config.device)
        embeddings = self.encoder.forward(features)
        # embeddings = self.vector_linear(embeddings)
        return embeddings
    
class EncodingModel_LLM2Vec_with_Reduction(nn.Module):
    def __init__(self, config, base_teacher):
        super().__init__()
        self.config = config
        self.dtype = torch.bfloat16 if config.dtype == 'bfloat16' else torch.float32
        
        # Store the pre-trained teacher
        self.base_teacher = base_teacher.encoder
        self.base_teacher.eval()
        # Freeze the base teacher
        # for param in self.base_teacher.parameters():
        #     param.requires_grad = False
            
        # Add reduction layer
        self.vector_linear = nn.Sequential(
            nn.Linear(4096, 768),  # From teacher size to student size
            nn.Tanh()
        ).to('cuda', dtype=self.dtype)

    def forward(self, inputs, is_des=False):
        # Get embeddings from frozen teacher
        with torch.set_grad_enabled(not self.training):
            features = self.base_teacher.tokenize(inputs)
            features = batch_to_device(features, self.config.device)
            teacher_embeddings = self.base_teacher.forward(features)
        
        # Apply reduction
        reduced_embeddings = self.vector_linear(teacher_embeddings)
        return reduced_embeddings, teacher_embeddings
    
if __name__ == '__main__':
    from config import Config
    config = Config('config.ini')
    model = EncodingModel_LLM2Vec(config)
    #inference
    inputs = ["hello world", 'papa i love you']
    embeddings = model(inputs)
    print(embeddings.size())