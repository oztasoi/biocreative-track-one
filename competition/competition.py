import sys
import json
import numpy as np

import torch as tr
import pandas as pd
import torch.nn as nn

import random as rn
from tqdm import trange
from numba import cuda
from GPUtil import showUtilization as gpu_usage
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import precision_score, f1_score, recall_score

label_dict = {
    'INHIBITOR': 0,
    'PART-OF': 1,
    'SUBSTRATE': 2,
    'ACTIVATOR': 3,
    'INDIRECT-DOWNREGULATOR': 4,
    'ANTAGONIST': 5,
    'INDIRECT-UPREGULATOR': 6,
    'AGONIST': 7,
    'DIRECT-REGULATOR': 8,
    'PRODUCT-OF': 9,
    'AGONIST-ACTIVATOR': 10,
    'AGONIST-INHIBITOR': 11,
    'SUBSTRATE_PRODUCT-OF': 12
}

rd_abs_split_tr = pd.read_csv("~/competition/training/drugprot_training_splitabs.tsv", sep="\t", header=None)
rd_ent_tr = pd.read_csv("~/competition/training/drugprot_training_entities.tsv", sep="\t", header=None)
rd_rel_tr = pd.read_csv("~/competition/training/drugprot_training_relations.tsv", sep="\t", header=None)

rd_ent_tr.columns = ["pubMedId", "entityId", "entityType", "sOffset", "eOffset", "entityText"]
rd_rel_tr.columns = ["pubMedId", "relType", "Arg1", "Arg2"]

rd_abs_split_dv = pd.read_csv("~/competition/development/drugprot_development_splitabs.tsv", sep="\t", header=None)
rd_ent_dv = pd.read_csv("~/competition/development/drugprot_development_entities.tsv", sep="\t", header=None)
rd_rel_dv = pd.read_csv("~/competition/development/drugprot_development_relations.tsv", sep="\t", header=None)

rd_ent_dv.columns = ["pubMedId", "entityId", "entityType", "sOffset", "eOffset", "entityText"]
rd_rel_dv.columns = ["pubMedId", "relType", "Arg1", "Arg2"]

def unify_abs(dataframe):
    np_matrix = dataframe.to_numpy()
    np_matrix = np_matrix[np.logical_not(pd.isnull(np_matrix))]
    np_list = list(np_matrix)
    abs_dict = dict()

    pm_id_list = list()
    for ix, val in enumerate(np_list):
        try:
            if int(val):
                pm_id_list.append(ix)
        except Exception:
            pass

    for i in range(len(pm_id_list)-1):
        sentence_list = np_list[pm_id_list[i]:pm_id_list[i+1]]
        abs_dict[int(sentence_list[0])] = sentence_list[1:]

    sentence_list = np_list[pm_id_list[-1]:]
    abs_dict[int(sentence_list[0])] = sentence_list[1:]
    return abs_dict

def find_sentence_index(start, sentence_list):
    for ix in range(1,len(sentence_list)):
        if int(start) <= len(" ".join(sentence_list[:ix])):
            return ix
    return ix

def calculate_prev_sentence_length(sentence_count, sentence_list):
    return len(" ".join(sentence_list[:sentence_count-1]))+1

class BioBertModel(nn.Module):
    def __init__(self):
        super(BioBertModel, self).__init__()
        self.model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        self.linear = nn.Linear(768, 13)

    def forward(self, tokens, masks=None):
        output = self.model(tokens, attention_mask=masks)[0]
        output = output[:,0,:]
        output = self.linear(output)
        return output

def preprocess(abstract_sentence_dict, entity_frame, relation_frame):
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    prepped_list = list()
    sentence_dict = {
        "input": None,
        "mask": None,
        "label": None,
        "arg1": None,
        "arg2": None
        }

    for pubMedId in abstract_sentence_dict.keys():
        chem_sentence_id_dict = dict()
        non_chem_sentence_id_dict = dict()
        entities = entity_frame.loc[entity_frame["pubMedId"] == pubMedId]
        sentence_list = abstract_sentence_dict[pubMedId]
        chem_entities = entities.loc[entities["entityType"] == "CHEMICAL"]
        non_chem_entities = entities.loc[entities["entityType"] != "CHEMICAL"]

        for chem_ix in range(chem_entities.shape[0]):
            sOffset = chem_entities.iloc[chem_ix]["sOffset"]
            entityId = chem_entities.iloc[chem_ix]["entityId"]
            sentence_count = find_sentence_index(sOffset, sentence_list)
            chem_sentence_id_dict[entityId] = {
                "sOffset": chem_entities.iloc[chem_ix]["sOffset"],
                "eOffset": chem_entities.iloc[chem_ix]["eOffset"],
                "sen_ct": sentence_count
                }

        for nonchem_ix in range(non_chem_entities.shape[0]):
            sOffset = non_chem_entities.iloc[nonchem_ix]["sOffset"]
            entityId = non_chem_entities.iloc[nonchem_ix]["entityId"]
            sentence_count = find_sentence_index(sOffset, sentence_list)
            non_chem_sentence_id_dict[entityId] = {
                "sOffset": non_chem_entities.iloc[nonchem_ix]["sOffset"],
                "eOffset": non_chem_entities.iloc[nonchem_ix]["eOffset"],
                "sen_ct": sentence_count
                }

        relations = relation_frame.loc[relation_frame["pubMedId"] == pubMedId]
        for relation_ix in range(relations.shape[0]):
            arg1 = relations.iloc[relation_ix]["Arg1"].split(":")[-1]
            arg2 = relations.iloc[relation_ix]["Arg2"].split(":")[-1]
            relType = relations.iloc[relation_ix]["relType"]

            type_arg1 = entities.loc[entities["entityId"] == arg1]["entityType"].to_string(index=False).strip()
            type_arg2 = entities.loc[entities["entityId"] == arg2]["entityType"].to_string(index=False).strip()

            if type_arg1 == "CHEMICAL":
                s1Offset = chem_sentence_id_dict[arg1]["sOffset"]
                e1Offset = chem_sentence_id_dict[arg1]["eOffset"]
                sentence_ct = chem_sentence_id_dict[arg1]["sen_ct"]
                rel_sentence = abstract_sentence_dict[pubMedId][sentence_ct-1]
                rel_sentence_offset = calculate_prev_sentence_length(sentence_ct, abstract_sentence_dict[pubMedId])
            else:
                s1Offset = non_chem_sentence_id_dict[arg1]["sOffset"]
                e1Offset = non_chem_sentence_id_dict[arg1]["eOffset"]
                sentence_ct = non_chem_sentence_id_dict[arg1]["sen_ct"]
                rel_sentence = abstract_sentence_dict[pubMedId][sentence_ct-1]
                rel_sentence_offset = calculate_prev_sentence_length(sentence_ct, abstract_sentence_dict[pubMedId])

            if type_arg2 == "CHEMICAL":
                s2Offset = chem_sentence_id_dict[arg2]["sOffset"]
                e2Offset = chem_sentence_id_dict[arg2]["eOffset"]
            else:
                s2Offset = non_chem_sentence_id_dict[arg2]["sOffset"]
                e2Offset = non_chem_sentence_id_dict[arg2]["eOffset"]

            if s2Offset < s1Offset:
                pre_arg2_text = tokenizer.encode(rel_sentence[:s2Offset-rel_sentence_offset], add_special_tokens=False)
                arg2_text = tokenizer.encode(rel_sentence[s2Offset-rel_sentence_offset:e2Offset-rel_sentence_offset], add_special_tokens=False)
                post_arg2_pre_arg1_text = tokenizer.encode(rel_sentence[e2Offset-rel_sentence_offset:s1Offset-rel_sentence_offset], add_special_tokens=False)
                arg1_text = tokenizer.encode(rel_sentence[s1Offset-rel_sentence_offset:e1Offset-rel_sentence_offset], add_special_tokens=False)
                post_arg1_text = tokenizer.encode(rel_sentence[e1Offset-rel_sentence_offset:], add_special_tokens=False)
                id_sentence = [101] + pre_arg2_text + [3] + arg2_text + [4] + post_arg2_pre_arg1_text + [1] + arg1_text + [2] + post_arg1_text + [102]
            else:
                pre_arg1_text = tokenizer.encode(rel_sentence[:s1Offset-rel_sentence_offset], add_special_tokens=False)
                arg1_text = tokenizer.encode(rel_sentence[s1Offset-rel_sentence_offset:e1Offset-rel_sentence_offset], add_special_tokens=False)
                post_arg1_pre_arg2_text = tokenizer.encode(rel_sentence[e1Offset-rel_sentence_offset:s2Offset-rel_sentence_offset], add_special_tokens=False)
                arg2_text = tokenizer.encode(rel_sentence[s2Offset-rel_sentence_offset:e2Offset-rel_sentence_offset], add_special_tokens=False)
                post_arg2_text = tokenizer.encode(rel_sentence[e2Offset-rel_sentence_offset:], add_special_tokens=False)
                id_sentence = [101] + pre_arg1_text + [1] + arg1_text + [2] + post_arg1_pre_arg2_text + [3] + arg2_text + [4] + post_arg2_text + [102]

            if len(id_sentence) < 512:
                sentence_dict = {
                    "input": id_sentence + [0] * (512 - len(id_sentence)),
                    "mask": [1] * len(id_sentence) + [0] * (512 - len(id_sentence)),
                    "label": label_dict[relType],
                    "arg1": arg1,
                    "arg2": arg2
                }
                prepped_list.append(sentence_dict)

    return prepped_list

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    tr.cuda.empty_cache()

    print("GPU Usage after emptying the cache")
    gpu_usage()

print("Training data extraction started.")
abstract_sentence_dict_tr = unify_abs(rd_abs_split_tr)
prepped_sentence_list_tr = preprocess(abstract_sentence_dict_tr, rd_ent_tr, rd_rel_tr)
print("Training data extraction completed.")

print("Test data extraction started.")
abstract_sentence_dict_dv = unify_abs(rd_abs_split_dv)
prepped_sentence_list_dv = preprocess(abstract_sentence_dict_dv, rd_ent_dv, rd_rel_dv)
print("Test data extraction completed.")

rn.seed(2021)
rn.shuffle(prepped_sentence_list_tr)
rn.shuffle(prepped_sentence_list_dv)

device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

training_sentences = prepped_sentence_list_tr
test_sentences = prepped_sentence_list_dv

BATCH_SIZE = 8
EPOCHS = 5

train_dataset = TensorDataset(
    tr.tensor([sentence["input"] for sentence in training_sentences]).to(device),
    tr.tensor([sentence["mask"] for sentence in training_sentences]).to(device),
    tr.tensor([sentence["label"] for sentence in training_sentences]).to(device)
    )
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(
    train_dataset,
    sampler=train_sampler,
    batch_size=BATCH_SIZE
    )

test_dataset = TensorDataset(
    tr.tensor([sentence["input"] for sentence in test_sentences]).to(device),
    tr.tensor([sentence["mask"] for sentence in test_sentences]).to(device),
    tr.tensor([sentence["label"] for sentence in test_sentences]).to(device)
    )
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(
    test_dataset,
    sampler=test_sampler,
    batch_size=BATCH_SIZE
    )

print("Training & Test data sets are created.")

lr_list = [1e-5, 3e-5, 5e-5]
wd_list = [1e-2, 3e-2, 5e-2]

fout = open("./eval.json", "w")

for lr in lr_list:
    for wd in wd_list:
        for model_instance in range(10):
            free_gpu_cache()
            bioBERT_model = BioBertModel()
            bioBERT_model = bioBERT_model.to(device)
            optimizer = tr.optim.Adam(params=bioBERT_model.parameters(), lr=lr, weight_decay=wd)

            bioBERT_model.train()
            loss_func = nn.CrossEntropyLoss()
            for epoch_num in trange(EPOCHS, desc="Epoch"):
                train_loss = 0.0
                for step_num, batch_data in enumerate(train_dataloader):
                    inputs, masks, labels = batch_data
                    outputs = bioBERT_model.forward(inputs, masks)
                    batch_loss = loss_func(outputs, labels)
                    train_loss += batch_loss.item()

                    batch_loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

            bioBERT_model.eval()
            all_predicted = []
            true_labels = []
            with tr.no_grad():
                for step_num, batch_data in enumerate(test_dataloader):

                    inputs, masks, labels = batch_data
                    outputs = bioBERT_model.forward(inputs, masks)
                    _, predicted = tr.max(outputs.data, 1)
                    predicted = predicted.tolist()

                    all_predicted += predicted
                    true_labels += labels.tolist()

                ps = precision_score(true_labels, all_predicted, average="micro")
                rs = recall_score(true_labels, all_predicted, average="micro")
                f1s = f1_score(true_labels, all_predicted, average="micro")

                json.dump({ "lr": lr, "wd": wd, "model_instance": model_instance,
                          "all_predicted": all_predicted,
                          "true_labels": true_labels,
                          "ps": ps, "rs": rs, "f1s": f1s }, fout)
                fout.flush()
            print(f"Iteration {model_instance} with lr: {lr} and wd: {wd}")
            free_gpu_cache()

fout.close()
print("Done!")
