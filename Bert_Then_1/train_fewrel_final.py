import argparse
import torch
import random
import sys
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from config import Config
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
import gc
import warnings
warnings.filterwarnings("ignore")


from sampler import data_sampler_CFRL
from data_loader import get_data_loader_BERT
from utils import Moment
from encoder import EncodingModel, EncodingModel_LLM2Vec, EncodingModel_LLM2Vec_with_Reduction
import wandb

from transformers import BertTokenizer
from losses import TripletLoss, OTLoss

class Manager(object):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        
    def _edist(self, x1, x2):
        '''
        input: x1 (B, H), x2 (N, H) ; N is the number of relations
        return: (B, N)
        '''
        b = x1.size()[0]
        L2dist = nn.PairwiseDistance(p=2)
        dist = [] # B
        for i in range(b):
            dist_i = L2dist(x2, x1[i])
            dist.append(torch.unsqueeze(dist_i, 0)) # (N) --> (1,N)
        dist = torch.cat(dist, 0) # (B, N)
        return dist
    def _cosine_similarity(self, x1, x2):
        '''
        input: x1 (B, H), x2 (N, H) ; N is the number of relations
        return: (B, N)
        '''
        b = x1.size()[0]
        cos = nn.CosineSimilarity(dim=1)
        sim = []
        for i in range(b):
            sim_i = cos(x2, x1[i])
            sim.append(torch.unsqueeze(sim_i, 0))
        sim = torch.cat(sim, 0)
        return sim
    

    def get_memory_proto(self, encoder, dataset, is_llm = False):
        '''
        only for one relation data
        '''
        data_loader = get_data_loader_BERT(config, dataset, shuffle=False, \
            drop_last=False,  batch_size=1) 
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                if isinstance(instance[k], list):
                    continue
                else:
                    instance[k] = instance[k].to(self.config.device)
            if is_llm:
                hidden = encoder(instance['input'])
            else:
                hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)    
        features = torch.cat(features, dim=0) # (M, H)
        proto = features.mean(0)

        return proto, features   


    def select_memory(self, encoder, dataset, is_llm = False):
        '''
        only for one relation data
        '''
        N, M = len(dataset), self.config.memory_size
        data_loader = get_data_loader_BERT(self.config, dataset, shuffle=False, \
            drop_last= False, batch_size=1) # batch_size must = 1
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                if isinstance(instance[k], list):
                    continue
                else:
                    instance[k] = instance[k].to(self.config.device)
            if is_llm:
                hidden = encoder(instance['input'])
            else:
                hidden = encoder(instance) 

            fea = hidden.detach().cpu().float().numpy()  # Convert to float32 and then to numpy array
            # fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)

        features = np.concatenate(features) # tensor-->numpy array; (N, H)
        
        if N <= M: 
            return copy.deepcopy(dataset), torch.from_numpy(features)

        num_clusters = M # memory_size < len(dataset)
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features) # (N, M)

        mem_set = []
        mem_feas = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            sample = dataset[sel_index]
            mem_set.append(sample)
            mem_feas.append(features[sel_index])

        mem_feas = np.stack(mem_feas, axis=0) # (M, H)
        mem_feas = torch.from_numpy(mem_feas)
        features = torch.from_numpy(features) # (N, H) tensor
        rel_proto = features.mean(0) # (H)

        return mem_set, mem_feas
        
    
    # def get_cluster_and_centroids(self, embeddings):
    #     # Before conversion to NumPy, convert to float32
    #     embeddings = embeddings.to(torch.float32).cpu().detach().numpy()
    #     clustering_model = AgglomerativeClustering(n_clusters=None,metric="cosine",linkage="average", distance_threshold=0.1).fit(embeddings)
    #     clusters = clustering_model.fit_predict(embeddings)
    #     centroids = {}
    #     for cluster_id in np.unique(clusters):
    #         if cluster_id not in centroids:
    #             cluster_embeddings = embeddings[clusters == cluster_id]
    #             centroid = torch.mean(cluster_embeddings, dim=0).to(torch.bfloat16 if self.config.dtype == 'bfloat16' else torch.float32)
    #             centroids[cluster_id] = centroid

    #     return clusters, centroids
    def get_cluster_and_centroids(self, embeddings):
    # Convert to NumPy for clustering
        embeddings_np = embeddings.to(torch.float32).cpu().detach().numpy()
        
        # Perform clustering on NumPy array
        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average", 
            distance_threshold=0.1
        ).fit(embeddings_np)
        
        clusters = clustering_model.fit_predict(embeddings_np)
        centroids = {}
        
        for cluster_id in np.unique(clusters):
            if cluster_id not in centroids:
                # Get cluster embeddings as NumPy array
                cluster_embeddings_np = embeddings_np[clusters == cluster_id]
                
                # Convert back to PyTorch tensor for mean calculation
                cluster_embeddings = torch.from_numpy(cluster_embeddings_np)
                
                # Calculate mean and convert to desired dtype
                centroid = torch.mean(cluster_embeddings, dim=0)
                centroid = centroid.to(torch.bfloat16 if self.config.dtype == 'bfloat16' else torch.float32)
                
                centroids[cluster_id] = centroid

        return clusters, centroids

    def train_model(self, encoder, teacher_with_reduction, training_data, seen_des, seen_relations, list_seen_des, is_memory=False):
        data_loader = get_data_loader_BERT(self.config, training_data, shuffle=True)

        optimizer = optim.Adam([
            {'params': encoder.parameters()}
            # {'params': teacher_with_reduction.vector_linear.parameters()}  # Only reduction layer
        ], lr=config.lr)

        encoder.train()
        teacher_with_reduction.eval()
        # teacher_with_reduction.vector_linear.train() 
        
        dtype = torch.bfloat16 if self.config.dtype == 'bfloat16' else torch.float32
        ot_loss = OTLoss(
            # input_dim=4096,
            # output_dim=768,
            distance_type='cosine',
            weight_type='norm',
            sinkhorn_epsilon=0.1,
            dtype=dtype,
            device=self.config.device
        )
        epoch = self.config.epoch_mem if is_memory else self.config.epoch

        triplet = TripletLoss()
        optimizer.zero_grad()

        relation_2_cluster = {}
        rep_seen_des = []
        relationid2_clustercentroids = {}
        
        accumulation_steps = self.config.accumulation_steps
        for i in range(epoch):         
            for batch_num, (instance, labels, ind) in enumerate(data_loader):
                for k in instance.keys():
                    if isinstance(instance[k], list):
                        continue
                    else:
                        instance[k] = instance[k].to(self.config.device)

                batch_instance = {'ids': [], 'mask': []} 

                batch_instance['ids'] = torch.tensor([seen_des[self.id2rel[label.item()]]['ids'] for label in labels]).to(self.config.device)
                batch_instance['mask'] = torch.tensor([seen_des[self.id2rel[label.item()]]['mask'] for label in labels]).to(self.config.device)

                
                hidden = encoder(instance) # b, dim
                teacher_reduced_output = teacher_with_reduction(instance['input'])
                rep_des = encoder(batch_instance, is_des = True) # b, dim

                with torch.no_grad():
                    rep_seen_des = []
                    for i2 in range(len(list_seen_des)):
                        sample = {
                            'ids' : torch.tensor([list_seen_des[i2]['ids']]).to(self.config.device),
                            'mask' : torch.tensor([list_seen_des[i2]['mask']]).to(self.config.device)
                        }
                        hidden_des = encoder(sample, is_des=True)
                        hidden_des = hidden_des.detach().cpu().data
                        rep_seen_des.append(hidden_des)
                    rep_seen_des = torch.cat(rep_seen_des, dim=0)
                    clusters, clusters_centroids = self.get_cluster_and_centroids(rep_seen_des)
                flag = 0
                if len(clusters) == max(clusters) + 1:
                    flag = 1

                # print(clusters)

                relationid2_clustercentroids = {}
                for index, rel in enumerate(seen_relations):
                    relationid2_clustercentroids[self.rel2id[rel]] = clusters_centroids[clusters[index]]

                relation_2_cluster = {}

                for i1 in range(len(seen_relations)):
                    relation_2_cluster[self.rel2id[seen_relations[i1]]] = clusters[i1]

                loss2 = self.moment.mutual_information_loss_cluster(hidden, rep_des, labels, temperature=0.01,relation_2_cluster=relation_2_cluster)  # Recompute loss2

                    
                cluster_centroids = []

                for label in labels:
                    cluster_centroids.append(relationid2_clustercentroids[label.item()])

                cluster_centroids  = torch.stack(cluster_centroids, dim = 0).to(self.config.device)
                
                nearest_cluster_centroids = []
                for hid in hidden:
                    cos_similarities = torch.nn.functional.cosine_similarity(hid.unsqueeze(0), cluster_centroids, dim=1)

                    try:
                        top2_similarities, top2_indices = torch.topk(cos_similarities, k=2, dim=0)

                        if len(top2_indices) > 1:
                            top2_centroids = relationid2_clustercentroids[labels[top2_indices[1].item()].item()]
                        else:
                            top2_centroids = relationid2_clustercentroids[labels[torch.argmax(cos_similarities).item()].item()]

                    except RuntimeError as e:
                        print(f"RuntimeError in top-k selection: {e}")
                        top2_centroids = relationid2_clustercentroids[labels[torch.argmax(cos_similarities).item()].item()]

                    nearest_cluster_centroids.append(top2_centroids)

                nearest_cluster_centroids = torch.stack(nearest_cluster_centroids, dim = 0).to(self.config.device)

                if flag == 0:
                    loss1 = self.moment.contrastive_loss(hidden, labels, is_memory, des =rep_des, relation_2_cluster = relation_2_cluster)

                    loss3 = triplet(hidden, rep_des, cluster_centroids)

                    loss4 = triplet(hidden, cluster_centroids, nearest_cluster_centroids)
                    
                    loss5 = ot_loss(hidden, teacher_reduced_output)

                    loss = loss1 + 1*loss2 + 0.25*loss3 + 0.25*loss4 + 4*loss5
                    
                    wandb.log({'OT_loss': loss5, 'contrastive_loss': loss1, 'total_loss': loss, 'MI_loss': loss2, 'triplet_loss': loss3, 'triplet_loss2': loss4})

                else:
                    loss1 = self.moment.contrastive_loss(hidden, labels, is_memory, des =rep_des, relation_2_cluster = relation_2_cluster)
                    loss5 = ot_loss(hidden, teacher_reduced_output)
                    loss = loss1 + 1*loss2 + 4*loss5
                    wandb.log({'OT_loss': loss5, 'contrastive_loss': loss1, 'total_loss': loss, 'MI_loss': loss2})
            
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Step optimization only after accumulating gradients
                if (batch_num + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                # update moment
                if is_memory:
                    self.moment.update_des(ind, hidden.detach().cpu().data, rep_des.detach().cpu().data, is_memory=True)
                    # self.moment.update_allmem(encoder)
                else:
                    self.moment.update_des(ind, hidden.detach().cpu().data, rep_des.detach().cpu().data, is_memory=False)
                # print
                if is_memory:
                    sys.stdout.write('MemoryTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()*accumulation_steps) + '\r')
                else:
                    sys.stdout.write('CurrentTrain: epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()*accumulation_steps) + '\r')
                sys.stdout.flush() 
                
            if (batch_num + 1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
                
        print('') 
        torch.cuda.empty_cache()
        gc.collect() 
        
    
                   
    def train_teacher(self, encoder_llm, training_data, seen_des, seen_relations, list_seen_des, is_memory=False):
        data_loader = get_data_loader_BERT(self.config, training_data, shuffle=True)
        optimizer_llm = optim.Adam(params=encoder_llm.parameters(), lr=self.config.lr)

        encoder_llm.train()
        epoch = self.config.epoch_mem if is_memory else self.config.epoch_llm

        triplet = TripletLoss()

        rep_seen_des = []
        relationid2_clustercentroids = {}

        accumulation_steps = self.config.accumulation_steps
        for i in range(epoch):
            
            for batch_num, (instance, labels, ind) in enumerate(data_loader):
                for k in instance.keys():
                    if isinstance(instance[k], list):
                        continue
                    else:
                        instance[k] = instance[k].to(self.config.device)
                batch_instance = {'input': []}
                batch_instance['input'] = [seen_des[self.id2rel[label.item()]]['input'][0] for label in labels]

                rep_seen_des = []
                with torch.no_grad(): 
                    for i2 in range(len(list_seen_des)):
                        sample = {
                            'input' : list_seen_des[i2]['input']
                        }
                        hidden = encoder_llm(sample['input'])
                        # hidden = hidden.detach().cpu()
                        rep_seen_des.append(hidden)
                    rep_seen_des = torch.cat(rep_seen_des, dim=0)
                    rep_seen_des = rep_seen_des.detach().cpu()

                hidden = encoder_llm(instance['input']) # b, dim
                rep_des = encoder_llm(batch_instance['input']) # b, di

                clusters, clusters_centroids = self.get_cluster_and_centroids(rep_seen_des)

                flag = 0
                if len(clusters) == max(clusters) + 1:
                    flag = 1

                relationid2_clustercentroids = {}
                for index, rel in enumerate(seen_relations):
                    relationid2_clustercentroids[self.rel2id[rel]] = clusters_centroids[clusters[index]]

                relation_2_cluster = {}
                for i1 in range(len(seen_relations)):
                    relation_2_cluster[self.rel2id[seen_relations[i1]]] = clusters[i1]

                loss2 = self.teacher_moment.mutual_information_loss_cluster(hidden, rep_des, labels, temperature=0.05,relation_2_cluster=relation_2_cluster)  # Recompute loss2

                cluster_centroids = []
                for label in labels:
                    cluster_centroids.append(relationid2_clustercentroids[label.item()])
                cluster_centroids  = torch.stack(cluster_centroids, dim = 0).to(self.config.device)
                
                nearest_cluster_centroids = []
                for hid in hidden:
                    cos_similarities = torch.nn.functional.cosine_similarity(hid.unsqueeze(0), cluster_centroids, dim=1)
                    try:
                        top2_similarities, top2_indices = torch.topk(cos_similarities, k=2, dim=0)

                        if len(top2_indices) > 1:
                            top2_centroids = relationid2_clustercentroids[labels[top2_indices[1].item()].item()]
                        else:
                            top2_centroids = relationid2_clustercentroids[labels[torch.argmax(cos_similarities).item()].item()]

                    except RuntimeError as e:
                        print(f"RuntimeError in top-k selection: {e}")
                        top2_centroids = relationid2_clustercentroids[labels[torch.argmax(cos_similarities).item()].item()]

                    nearest_cluster_centroids.append(top2_centroids)

                nearest_cluster_centroids = torch.stack(nearest_cluster_centroids, dim = 0).to(self.config.device)
                if flag == 0:
                    loss1 = self.teacher_moment.contrastive_loss(hidden, labels, is_memory, des =rep_des, relation_2_cluster = relation_2_cluster)

                    loss3 = triplet(hidden, rep_des,  cluster_centroids)

                    loss4 = triplet(hidden, cluster_centroids, nearest_cluster_centroids)

                    loss = loss1 + 2*loss2 + 0.5*loss3 + 0.5*loss4
                    
                    wandb.log({'teacher_contrastive_loss': loss1, 'teacher_total_loss': loss, 'teacher_MI_loss': loss2, 'teacher_triplet_loss': loss3, 'teacher_triplet_loss2': loss4})
                else:
                    loss1 = self.teacher_moment.contrastive_loss(hidden, labels, is_memory, des =rep_des, relation_2_cluster = relation_2_cluster)

                    loss = loss1 + 2*loss2   
                    
                    wandb.log({'teacher_contrastive_loss': loss1, 'teacher_total_loss': loss, 'teacher_MI_loss': loss2})
    
                    
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()

                # Step optimization only after accumulating gradients
                if (batch_num + 1) % accumulation_steps == 0:
                    optimizer_llm.step()
                    optimizer_llm.zero_grad()
                    
                # update moment
                if is_memory:
                    self.teacher_moment.update_des(ind, hidden.detach().cpu().data, rep_des.detach().cpu().data, is_memory=True)
                else:
                    self.teacher_moment.update_des(ind, hidden.detach().cpu().data, rep_des.detach().cpu().data, is_memory=False)
                # print
                if is_memory:
                    sys.stdout.write('MemoryTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()*accumulation_steps) + '\r')
                else:
                    sys.stdout.write('CurrentTrain: epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()*accumulation_steps) + '\r')
                sys.stdout.flush() 
            
            # Handle any remaining gradients at the end of epoch
            if (batch_num + 1) % accumulation_steps != 0:
                optimizer_llm.step()
                optimizer_llm.zero_grad()
        
   
        print('')   
        torch.cuda.empty_cache()
        gc.collect()
                
               
    def eval_encoder_proto(self, encoder, seen_proto, seen_relid, test_data):
        batch_size = 16
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)
        
        corrects = 0.0
        total = 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                if isinstance(instance[k], list):
                    continue
                else:
                    instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance)
            fea = hidden.cpu().data # place in cpu to eval
            logits = -self._edist(fea, seen_proto) # (B, N) ;N is the number of seen relations

            cur_index = torch.argmax(logits, dim=1) # (B)
            pred =  []
            for i in range(cur_index.size()[0]):
                pred.append(seen_relid[int(cur_index[i])])
            pred = torch.tensor(pred)

            correct = torch.eq(pred, label).sum().item()
            acc = correct / batch_size
            corrects += correct
            total += batch_size
            sys.stdout.write('[EVAL] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   '\
                .format(batch_num, 100 * acc, 100 * (corrects / total)) + '\r')
            sys.stdout.flush()        
        print('')
        return corrects / total
    
    def eval_encoder_proto_des(self, encoder, seen_proto, seen_relid, test_data, rep_des, is_llm= False):
        """
        Args:
            encoder: Encoder
            seen_proto: seen prototypes. NxH tensor
            seen_relid: relation id of protoytpes
            test_data: test data
            rep_des: representation of seen relation description. N x H tensor

        Returns:

        """
        batch_size = 16
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)

        corrects = 0.0
        corrects1 = 0.0
        corrects2 = 0.0
        total = 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                if isinstance(instance[k], list):
                    continue
                else:
                    instance[k] = instance[k].to(self.config.device)
            with torch.no_grad():
                if is_llm:
                    hidden = encoder(instance['input'])
                else:
                    hidden = encoder(instance)
            fea = hidden.cpu().data  # place in cpu to eval
            # logits = -self._edist(fea, seen_proto)  # (B, N) ;N is the number of seen relations
            logits = self._cosine_similarity(fea, seen_proto)  # (B, N)
            logits_des = self._cosine_similarity(fea, rep_des)  # (B, N)

            logits_rrf = logits + logits_des 
           
            cur_index = torch.argmax(logits, dim=1)  # (B)
            pred = []
            for i in range(cur_index.size()[0]):
                pred.append(seen_relid[int(cur_index[i])])
            pred = torch.tensor(pred)
            correct = torch.eq(pred, label).sum().item()
            acc = correct / batch_size
            corrects += correct
            total += batch_size

            # by logits_des
            cur_index1 = torch.argmax(logits_des,dim=1)
            pred1 = []
            for i in range(cur_index1.size()[0]):
                pred1.append(seen_relid[int(cur_index1[i])])
            pred1 = torch.tensor(pred1)
            correct1 = torch.eq(pred1, label).sum().item()
            acc1 = correct1/ batch_size
            corrects1 += correct1

            # by rrf
            cur_index2 = torch.argmax(logits_rrf,dim=1)
            pred2 = []
            for i in range(cur_index2.size()[0]):
                pred2.append(seen_relid[int(cur_index2[i])])
            pred2 = torch.tensor(pred2)
            correct2 = torch.eq(pred2, label).sum().item()
            acc2 = correct2/ batch_size
            corrects2 += correct2

            

            sys.stdout.write('[EVAL] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   ' \
                             .format(batch_num, 100 * acc, 100 * (corrects / total)) + '\r')
            sys.stdout.write('[EVAL DES] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   ' \
                             .format(batch_num, 100 * acc1, 100 * (corrects1 / total)) + '\r')
            sys.stdout.write('[EVAL RRF] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   ' \
                             .format(batch_num, 100 * acc2, 100 * (corrects2 / total)) + '\r')
            sys.stdout.flush()
        print('')
        return corrects / total, corrects1 / total, corrects2 / total

    def _get_sample_text(self, data_path, index):
        sample = {}
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i == index:
                    items = line.strip().split('\t')
                    sample['relation'] = self.id2rel[int(items[0])-1]
                    sample['tokens'] = items[2]
                    sample['h'] = items[3]
                    sample['t'] = items[5]
        return sample

    def _read_description(self, r_path):
        rset = {}
        with open(r_path, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                rset[items[1]] = items[2]
        return rset

    
    def train(self):
        # sampler 
        sampler = data_sampler_CFRL(config=self.config, seed=self.config.seed)
        self.config.vocab_size = sampler.config.vocab_size

        print('prepared data!')
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id
        self.r2desc = self._read_description(self.config.relation_description)

        # encoder
        encoder = EncodingModel(self.config)
        teacher = EncodingModel_LLM2Vec(self.config)

        # step is continual task number
        cur_acc, total_acc = [], []
        cur_acc1, total_acc1 = [], []
        cur_acc2, total_acc2 = [], []
        
        cur_acc_llm, total_acc_llm = [], []
        cur_acc1_llm, total_acc1_llm = [], []
        cur_acc2_llm, total_acc2_llm = [], []


        cur_acc_num, total_acc_num = [], []
        cur_acc_num1, total_acc_num1 = [], []
        cur_acc_num2, total_acc_num2 = [], []
        
        cur_acc_num_llm, total_acc_num_llm = [], []
        cur_acc_num1_llm, total_acc_num1_llm = [], []
        cur_acc_num2_llm, total_acc_num2_llm = [], []


        memory_samples = {}
        memory_samples_llm = {}
        data_generation = []
        seen_des = {}


        self.unused_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
        self.unused_token = '[unused0]'
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, \
            additional_special_tokens=[self.unused_token])


        for step, (training_data, valid_data, test_data, current_relations, \
            historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):

            for rel in current_relations:
                ids = self.tokenizer.encode(seen_descriptions[rel][0],
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.config.max_length)        
                # mask
                mask = np.zeros(self.config.max_length, dtype=np.int32)
                end_index = np.argwhere(np.array(ids) == self.tokenizer.get_vocab()[self.tokenizer.sep_token])[0][0]
                mask[:end_index + 1] = 1 
                if rel not in seen_des:
                    seen_des[rel] = {}
                    seen_des[rel]['ids'] = ids
                    seen_des[rel]['mask'] = mask
                    seen_des[rel]['input'] = seen_descriptions[rel]

            # get representation of seen description
            seen_relid = []
            for rel in seen_relations:
                seen_relid.append(self.rel2id[rel])

            seen_des_by_id = {}
            for rel in seen_relations:
                seen_des_by_id[self.rel2id[rel]] = seen_des[rel]

            list_seen_des = []
            for i in range(len(seen_relations)):
                list_seen_des.append(seen_des_by_id[seen_relid[i]])

            # Initialization
            self.moment = Moment(self.config)
            self.teacher_moment = Moment(self.config)   

            # Train current task
            training_data_initialize = []

            if step > 0:
                relations = list(set(seen_relations) - set(current_relations))
                for rel in relations:
                    training_data_initialize += memory_samples[rel]            

            for rel in current_relations:
                training_data_initialize += training_data[rel]
            
            self.teacher_moment.init_moment(teacher, training_data_initialize, seen_des, self.id2rel, is_memory=False, is_llm=True)
            self.train_teacher(teacher, training_data_initialize, seen_des, seen_relations, list_seen_des, is_memory=False)

            self.moment.init_moment(encoder, training_data_initialize, seen_des, self.id2rel, is_memory=False, is_llm=False)
            # teacher_with_reduction = EncodingModel_LLM2Vec_with_Reduction(self.config, teacher)
            self.train_model(encoder, teacher, training_data_initialize, seen_des, seen_relations, list_seen_des, is_memory=False)


            # Select memory samples
            for rel in current_relations:
                memory_samples_llm[rel], _ = self.select_memory(teacher, training_data[rel], is_llm = True)
                memory_samples[rel], _ = self.select_memory(encoder, training_data[rel])
                
                    
            # Update proto
            seen_proto = []  
            seen_proto_llm = []  
            for rel in seen_relations:
                proto_llm, _ = self.get_memory_proto(teacher, memory_samples_llm[rel], is_llm = True)
                proto, _ = self.get_memory_proto(encoder, memory_samples[rel])
                seen_proto.append(proto)
                seen_proto_llm.append(proto_llm)
            seen_proto = torch.stack(seen_proto, dim=0)
            seen_proto_llm = torch.stack(seen_proto_llm, dim=0)

            # get seen relation id
            seen_relid = []
            for rel in seen_relations:
                seen_relid.append(self.rel2id[rel])

            # Eval current task and history task
            test_data_initialize_cur, test_data_initialize_seen = [], []
            for rel in current_relations:
                test_data_initialize_cur += test_data[rel]
            for rel in seen_relations:
                test_data_initialize_seen += historic_test_data[rel]
            
            with torch.no_grad():
                encoder.eval()
                teacher.eval()
                rep_des = []
                rep_des_llm = []
                for i in range(len(list_seen_des)):
                    sample = {
                        'ids' : torch.tensor([list_seen_des[i]['ids']]).to(self.config.device),
                        'mask' : torch.tensor([list_seen_des[i]['mask']]).to(self.config.device)
                    }
                    hidden = encoder(sample, is_des=True)
                    hidden = hidden.detach().cpu().data
                    rep_des.append(hidden)
                    
                    sample_llm = {
                            'input' : [list_seen_des[i]['input'][0]]
                        }
                    hidden_llm = teacher(sample_llm['input'])
                    hidden_llm = hidden_llm.detach().cpu().data
                    rep_des_llm.append(hidden_llm)
                    
                    
                rep_des = torch.cat(rep_des, dim=0)
                rep_des_llm = torch.cat(rep_des_llm, dim=0)
            encoder.train()
            teacher.train()

            ac1,ac1_des, ac1_rrf = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_cur,rep_des)
            ac2,ac2_des, ac2_rrf = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_seen, rep_des)
            
            ac1_llm, ac1_des_llm, ac1_rrf_llm = self.eval_encoder_proto_des(teacher, seen_proto_llm, seen_relid, test_data_initialize_cur, rep_des_llm, is_llm= True)
            ac2_llm, ac2_des_llm, ac2_rrf_llm = self.eval_encoder_proto_des(teacher, seen_proto_llm, seen_relid, test_data_initialize_seen, rep_des_llm, is_llm= True)
            
            cur_acc_num.append(ac1)
            total_acc_num.append(ac2)
            cur_acc.append('{:.4f}'.format(ac1))
            total_acc.append('{:.4f}'.format(ac2))
            print('cur_acc: ', cur_acc)
            print('his_acc: ', total_acc)

            cur_acc_num1.append(ac1_des)
            total_acc_num1.append(ac2_des)
            cur_acc1.append('{:.4f}'.format(ac1_des))
            total_acc1.append('{:.4f}'.format(ac2_des))
            print('cur_acc des: ', cur_acc1)
            print('his_acc des: ', total_acc1)

            cur_acc_num2.append(ac1_rrf)
            total_acc_num2.append(ac2_rrf)
            cur_acc2.append('{:.4f}'.format(ac1_rrf))
            total_acc2.append('{:.4f}'.format(ac2_rrf))
            print('cur_acc rrf: ', cur_acc2)
            print('his_acc rrf: ', total_acc2)
            
            cur_acc_num_llm.append(ac1_llm)
            total_acc_num_llm.append(ac2_llm)
            cur_acc_llm.append('{:.4f}'.format(ac1_llm))
            total_acc_llm.append('{:.4f}'.format(ac2_llm))
            print('cur_acc llm: ', cur_acc_llm)
            print('his_acc llm: ', total_acc_llm)
            
            cur_acc_num1_llm.append(ac1_des_llm)
            total_acc_num1_llm.append(ac2_des_llm)
            cur_acc1_llm.append('{:.4f}'.format(ac1_des_llm))
            total_acc1_llm.append('{:.4f}'.format(ac2_des_llm))
            print('cur_acc des llm: ', cur_acc1_llm)
            print('his_acc des llm: ', total_acc1_llm)
            
            cur_acc_num2_llm.append(ac1_rrf_llm)
            total_acc_num2_llm.append(ac2_rrf_llm)
            cur_acc2_llm.append('{:.4f}'.format(ac1_rrf_llm))
            total_acc2_llm.append('{:.4f}'.format(ac2_rrf_llm))
            print('cur_acc rrf llm: ', cur_acc2_llm)
            print('his_acc rrf llm: ', total_acc2_llm)
            
            
            wandb.log({'step': step, 'cur_acc': ac1, 'his_acc': ac2, 'cur_acc_des': ac1_des, 'his_acc_des': ac2_des, 'cur_acc_rrf': ac1_rrf, 'his_acc_rrf': ac2_rrf, 'cur_acc_llm': ac1_llm, 'his_acc_llm': ac2_llm, 'cur_acc_des_llm': ac1_des_llm, 'his_acc_des_llm': ac2_des_llm, 'cur_acc_rrf_llm': ac1_rrf_llm, 'his_acc_rrf_llm': ac2_rrf_llm})


        torch.cuda.empty_cache()
        gc.collect()
        # save model
        # torch.save(encoder.state_dict(), "./checkpoints/encoder.pth")
        return total_acc_num, total_acc_num1, total_acc_num2, total_acc_num_llm, total_acc_num1_llm, total_acc_num2_llm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="FewRel", type=str)
    parser.add_argument("--num_k", default=5, type=int)
    parser.add_argument("--num_gen", default=2, type=int)
    args = parser.parse_args()
    config = Config('config.ini')
    config.task_name = args.task_name
    config.num_k = args.num_k
    config.num_gen = args.num_gen
    
    wandb.init(
    project = 'CPL-Bert-Then',
    name = f"CPL{args.task_name}_{args.num_k}-shot-SIRUS-Bert-Then",
    config = {
        'name': "CPL-SIRUS-Bert-Then",
        "task" : args.task_name,
        "shot" : f"{args.num_k}-shot"
    }
        )

    # config 
    print('#############params############')
    print(config.device)
    config.device = torch.device(config.device)
    print(f'Task={config.task_name}, {config.num_k}-shot')
    print(f'Encoding model: {config.model}')
    print(f'pattern={config.pattern}')
    print(f'mem={config.memory_size}, margin={config.margin}, gen={config.gen}, gen_num={config.num_gen}')
    print('#############params############')

    if config.task_name == 'FewRel':
        config.rel_index = './data/CFRLFewRel/rel_index.npy'
        config.relation_name = './data/CFRLFewRel/relation_name.txt'
        config.relation_description = './data/CFRLFewRel/relation_description.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/test_0.txt'
    else:
        config.rel_index = './data/CFRLTacred/rel_index.npy'
        config.relation_name = './data/CFRLTacred/relation_name.txt'
        config.relation_description = './data/CFRLTacred/relation_description.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_5/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_5/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_10/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_10/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_10/test_0.txt'        

    # seed 
    random.seed(config.seed) 
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)   
    base_seed = config.seed

    acc_list = []
    acc_list1 = []
    aac_list2 = []
    acc_list_llm = []
    acc_list1_llm = []
    aac_list2_llm = []
    for i in range(config.total_round):
        config.seed = base_seed + i * 100
        print('--------Round ', i)
        print('seed: ', config.seed)
        manager = Manager(config)
        acc, acc1, aac2, acc_llm, acc1_llm, aac2_llm = manager.train()
        acc_list.append(acc)
        acc_list1.append(acc1)
        aac_list2.append(aac2)
        acc_list_llm.append(acc_llm)
        acc_list1_llm.append(acc1_llm)
        aac_list2_llm.append(aac2_llm)
        torch.cuda.empty_cache()
        gc.collect()
    
    accs = np.array(acc_list)
    ave = np.mean(accs, axis=0)
    print('----------END')
    print('his_acc mean: ', np.around(ave, 4))
    accs1 = np.array(acc_list1)
    ave1 = np.mean(accs1, axis=0)
    print('his_acc des mean: ', np.around(ave1, 4))
    accs2 = np.array(aac_list2)
    ave2 = np.mean(accs2, axis=0)
    print('his_acc rrf mean: ', np.around(ave2, 4))
    
    accs_llm = np.array(acc_list_llm)
    ave_llm = np.mean(accs_llm, axis=0)
    print('his_acc_llm mean: ', np.around(ave_llm, 4))
    accs1_llm = np.array(acc_list1_llm)
    ave1_llm = np.mean(accs1_llm, axis=0)
    print('his_acc des_llm mean: ', np.around(ave1_llm, 4))
    accs2_llm = np.array(aac_list2_llm)
    ave2_llm = np.mean(accs2_llm, axis=0)
    print('his_acc rrf_llm mean: ', np.around(ave2_llm, 4))
    
    
    for i in range(len(ave)):
        wandb.log({f'his_acc_mean': ave[i], f'his_acc_des_mean': ave1[i], f'his_acc_rrf_mean': ave2[i], f'his_acc_llm_mean': ave_llm[i], f'his_acc_des_llm_mean': ave1_llm[i], f'his_acc_rrf_llm_mean': ave2_llm[i]})
    