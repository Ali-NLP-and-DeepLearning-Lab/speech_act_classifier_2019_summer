import pdb
import os
import csv
import nltk
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer

NO_CODE = "ZZ"


val_set = ['D121', 'D127', 'D126', 'D125', 'D124',
            'DYAD26', 'DYAD02', 'DYAD25', 'DYAD22', 'DYAD05', 'DYAD20', 'DYAD07']
def parse_file(f):

    lines = []
    corpus = {}

    while True:
        line = f.readline()
        if not line :
            break
        
        if line[0]=='D':
            lines = []
            dialogue_id= (line.split(' ')[0])
        elif line[0] == ' ':
            lines.append(line.strip())
        elif line[0] == 'Z':
            if dialogue_id in corpus.keys():
                corpus[dialogue_id].append(lines)
            else:
                corpus[dialogue_id] = [lines]

    
    return corpus


def parse_lines(lines_list):

    ts_dict = {}
    for lines in lines_list:
        for line in lines:
            segments = line.split(' ')
            for i in range(len(segments)/3):
                ts_id = segments[i*3]+ ' ' + segments[i*3+1]
                mode = segments[i*3+2]
                if ts_id in ts_dict.keys():
                    if mode not in ts_dict[ts_id].split('/'):
                        ts_dict[ts_id] += '/'+mode
                else:
                    ts_dict[ts_id] = mode
    
    return ts_dict
                
def extract_paranthesis(utterance):
    if '(' in utterance and ')' in utterance:
        #return utterance[utterance.find('(')+1:utterance.find(')')]
        
        return re.sub( '\(.*\)','',utterance)
    else :
        return utterance


def write_csv_file_train(corpus_list, train_save_path, data_dir='./data'):
    
    csv_f_train = open(train_save_path,mode='w')
    wr_train = csv.writer(csv_f_train)
    wr_train.writerow(['conversation_id','utterance_id','utterance','mode','caller'])
    '''
    csv_f_val = open(val_save_path,mode='w')
    wr_val = csv.writer(csv_f_val)
    wr_val.writerow(['conversation_id','utterance_id','utterance','mode','caller'])
    '''
    no_code_count = 0
    total_code = 0
    dl_count = 0
    for corpus in corpus_list:
        
        for dialogue_id in corpus.keys():

            if dialogue_id in val_set:
                #wr = wr_val
                continue
            else :
                wr = wr_train
            lines_list = corpus[dialogue_id]
            data_path = os.path.join(data_dir,dialogue_id+'.VRM')

            ts_dict = parse_lines(lines_list)

            f = open(data_path, 'r')
        
            utt_count = 0
            while True:
                line = f.readline()
                if not line:
                    break

                ts_id = line[:7]
                if ts_id[:3] == 'END' or len(ts_id)<3:
                    continue


                if dialogue_id[:4] == 'DYAD':
                    utterance = line[8:].strip()#.lower()
                else:
                    utterance = line[11:].strip()#.lower()

                utterance = extract_paranthesis(utterance)
                utterance = re.sub('/', ' ', utterance)
                tokens = nltk.word_tokenize(utterance)
                tokens = nltk.pos_tag(tokens)
                
                tokens = ['/'.join((token[0].lower(),token[1])) for token in tokens if len(token) == 2]
                utterance = ' '.join(tokens) #TreebankWordDetokenizer().detokenize(tokens)

                # process utterance
                if ts_id in ts_dict.keys():

                    wr.writerow([dl_count,utt_count,utterance,ts_dict[ts_id],ts_id[-1]])
                else :
                    wr.writerow([dl_count,utt_count,utterance,NO_CODE,ts_id[-1]])
                    no_code_count +=1
                utt_count += 1
                total_code += 1
            dl_count += 1

    print(total_code)
    print(no_code_count)
    
    return 



def write_csv_file_val(corpus_list, val_save_path, data_dir='./data'):
    
    
    csv_f_val = open(val_save_path,mode='w')
    wr_val = csv.writer(csv_f_val)
    wr_val.writerow(['conversation_id','utterance_id','utterance','mode','caller'])
    
    no_code_count = 0
    total_code = 0
    dl_count = 0
    max_utt = -1
    for corpus in corpus_list:
        
        for dialogue_id in corpus.keys():

            if dialogue_id in val_set:
                wr = wr_val
            else :
                #wr = wr_train
                continue
            lines_list = corpus[dialogue_id]
            data_path = os.path.join(data_dir,dialogue_id+'.VRM')

            ts_dict = parse_lines(lines_list)

            f = open(data_path, 'r')
        
            utt_count = 0
            while True:
                line = f.readline()
                if not line:
                    break

                ts_id = line[:7]
                if ts_id[:3] == 'END' or len(ts_id)<3:
                    continue


                if dialogue_id[:4] == 'DYAD':
                    utterance = line[8:].strip()#.lower()
                else:
                    utterance = line[11:].strip()#.lower()

                utterance = extract_paranthesis(utterance)
                utterance = re.sub('/', ' ', utterance)
                tokens = nltk.word_tokenize(utterance)
                tokens = nltk.pos_tag(tokens)
                
                tokens = ['/'.join((token[0].lower(),token[1])) for token in tokens if len(token) == 2]
                utterance = ' '.join(tokens) #TreebankWordDetokenizer().detokenize(tokens)

                # process utterance
                if ts_id in ts_dict.keys():

                    wr.writerow([dl_count,utt_count,utterance,ts_dict[ts_id],ts_id[-1]])
                else :
                    wr.writerow([dl_count,utt_count,utterance,NO_CODE,ts_id[-1]])
                    no_code_count +=1
                utt_count += 1
                total_code += 1
            dl_count += 1
            
    print(total_code)
    print(no_code_count)
    
    return 

        




f0 = open('./data/TENGVRMS.DAT','r')
f1 = open('./data/EMP2VRMS.DAT','r')

corpus_list = [parse_file(f0),parse_file(f1)]

x = write_csv_file_train(corpus_list,'vrm_train_data.csv')
x = write_csv_file_val(corpus_list,'vrm_val_data.csv')



