from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import nltk
import re
import numpy as np
import torch
import torch.nn.functional as F
import pdb
import os,sys
sys.path.insert(0, os.path.dirname(os.getcwd()))
from src.utils.word2vec import word2vec
from src.model.bilstm_ram import build_bilstm_ram as build_model_ram
from src.config import bilstm_ram_config, bilstm_ram_vrm_config

parent_dir = os.path.dirname(os.getcwd())
embedding_path = os.path.join(parent_dir,'data','word2vec_from_glove.bin')
sent_len = bilstm_ram_config.sent_len
assert(bilstm_ram_vrm_config.sent_len==sent_len)

pos_tag_len = 45
print("getting embedding...")
wv = word2vec(embedding_path,sent_len=sent_len)
_words_to_vector = wv.to_vector
print("done..!")

print("setting models...")

use_cuda = False
device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

model_swda = build_model_ram(device,bilstm_ram_config)
model_vrm = build_model_ram(device,bilstm_ram_vrm_config)
model_swda = model_swda.to(device)
model_vrm = model_vrm.to(device)

model_swda.load_state_dict(torch.load(os.path.join(parent_dir,'trained_model','trained_model_SwDA.pth')))
model_vrm.load_state_dict(torch.load(os.path.join(parent_dir,'trained_model','trained_model_VRM.pth')))

model_swda.eval()
model_vrm.eval()

print("done...!")


top_num = 3

@csrf_exempt
def classify_swda(request):

    json_data = json.loads(request.body)
    data = preprocess(json_data['dialogs'])

    data = [d.to(device).unsqueeze(0) for d in data]


    preds = model_swda(data)


    results = {'results':[]}

    for pred in preds:
        pred = pred.squeeze(0)
        pred = F.softmax(pred,-1)

        result = {}

        prob,label_indx = pred.max(-1)
        #print(label_indx.item())

        top_index = pred.argsort(descending=True)[0:top_num]
        for label_index in top_index:
            result[index_to_label_SWDA[label_index]] = pred[label_index].item()

        results['results'].append(result)


    return JsonResponse(results)
@csrf_exempt
def classify_vrm(request):

    json_data = json.loads(request.body)
    data = preprocess(json_data['dialogs'])

    data = [d.to(device).unsqueeze(0) for d in data]


    preds = model_vrm(data)


    results = {'results':[]}

    for pred in preds:
        pred = pred.squeeze(0)
        result = {}

        for axis in range(6):
            _pred = pred[2*axis:2*axis+2]
            _pred = F.softmax(_pred,-1)
            _pred = _pred[-1]
            result[axis_name[axis]] = _pred.item()

        results['results'].append(result)


    return JsonResponse(results)

def preprocess(dialogs):

    previous_caller = None

    utterance_chunk = []
    pos_chunk = []
    mask_chunk = []
    qtag_chunk = []
    caller_chunk = []

    for message_caller in dialogs:

        utterance = message_caller['message'].strip()
        utterance = re.sub('[()/\\\{}-]',' ',utterance)
        if utterance[-1] == '?' :
            qtag = 1
        else :
            qtag = 0

        tokens = nltk.word_tokenize(utterance)
        tokens = nltk.pos_tag(tokens)
        words = [token[0] for token in tokens if len(token) == 2]
        true_length = len(words)
        words = _words_to_vector(words)
        poses = [token[1] for token in tokens if len(token) == 2]
        poses =_poses_to_vector(poses)

        if previous_caller and message_caller['caller'] == previous_caller:
            caller = 1
        else:
            caller = 0
        previous_caller = message_caller['caller']

        mask = [ 1 for _ in range(true_length)][:sent_len] + [0 for _ in range(sent_len-true_length)]

        utterance_chunk.append(words)
        pos_chunk.append(poses)
        mask_chunk.append(mask)
        qtag_chunk.append(qtag)
        caller_chunk.append(caller)

    utterance_chunk = torch.stack(utterance_chunk)
    pos_chunk = torch.tensor(pos_chunk).to(utterance_chunk.dtype)
    mask_chunk = torch.tensor(mask_chunk).to(utterance_chunk.dtype)
    qtag_chunk = torch.tensor(qtag_chunk).to(utterance_chunk.dtype)
    caller_chunk = torch.tensor(caller_chunk).to(utterance_chunk.dtype)

    return utterance_chunk, pos_chunk,mask_chunk, qtag_chunk,caller_chunk

def _poses_to_vector(poses):
    pos_index = [pos_map[p] for p in poses]

    pos_vector = np.zeros((sent_len,pos_tag_len))
    pos_index = pos_index[:sent_len]

    for i,j in enumerate(pos_index):
        pos_vector[i][j]=1

    return pos_vector

pos_map = {'LS': 0, 'TO': 1, 'VBN': 2, "''": 3, 'WP': 4, 'UH': 5, 'VBG': 6, 'JJ': 7, 'VBZ': 8, '--': 9,
    'VBP': 10, 'NN': 11, 'DT': 12, 'PRP': 13, ':': 14, 'WP$': 15, 'NNPS': 16, 'PRP$': 17, 'WDT': 18,
    '(': 19, ')': 20, '.': 21, ',': 22, '``': 23, '$': 24, 'RB': 25, 'RBR': 26, 'RBS': 27, 'VBD': 28,
    'IN': 29, 'FW': 30, 'RP': 31, 'JJR': 32, 'JJS': 33, 'PDT': 34, 'MD': 35, 'VB': 36, 'WRB': 37,
    'NNP': 38, 'EX': 39, 'NNS': 40, 'SYM': 41, 'CC': 42, 'CD': 43, 'POS': 44}

label_map_SWDA = {'b': 0, 'fo': 1, 'ny': 2, '%': 3, 'sv': 4, 'qy^d': 5, 'aa': 6, 'qh': 7, 'ba': 8, 'nn': 9, \
    'ad': 10, 't3': 11, 't1': 12, 'ft': 13, 'bf': 14, 'qy': 15, 'ng': 16, 'no': 17, 'b^m': 18, 'h': 19, \
        '^2': 20, 'br': 21, 'qw^d': 22, 'bh': 23, 'qo': 24, 'fp': 25, 'sd': 26, 'na': 27, 'x': 28, 'qw': 29, \
            'fc': 30, 'bd': 31, 'oo': 32, 'fa': 33, '^g': 34, 'arp/nd': 35, '^h': 36, 'aap/am': 37, 'ar': 38, \
                'bk': 39, '^q': 40, 'qrr': 41}

index_to_label_SWDA = [label for label in label_map_SWDA.keys()]

axis_name = ['form_SE', 'form_PE', 'form_FR', 'intent_SE', 'intent_PE', 'intent_FR']
