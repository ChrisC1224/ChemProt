import collections
import json
import os
import random
import re
import string

import numpy as np
import pandas as pd
import progressbar
from geniatagger import GeniaTagger
from gensim.models import KeyedVectors


class Preprocessor(object):
    '''a collection of preprocessing operations.
    '''
    extra_token_list = ['CHEMICAL1', 'GENE1', 'CHEMICALN',
                        'GENEN', 'NUM', 'UNKNOWN']
    
    def __init__(self):
        '''Init the preprocessor.
        '''
        # self.train_list, self.test_list = self.get_all_data_list()
        self.entity_dict = {}


    def load_pretrained_emb(self):
        '''Load the pre-trained word embedding.
        
        '''
        pretrained_emb_path = os.getenv('pretrained_emb_path')
        binary = pretrained_emb_path.endswith('.bin')
        word_vectors = KeyedVectors.load_word2vec_format(pretrained_emb_path,
                                                         binary=binary)
        self.pretrained_emb = word_vectors

    def Extract_Sentences(self):
        '''Extract all sentences in a paper to a list

        Return:
             sen_list: a sentence list
             abs_dic: a abstracts dictionary
        '''
        abstracts_columns_list = ['PMID', 'Title', 'Abstract']
        # Training
        # training_abstracts_path = os.getenv('training_abstracts_path')
        # D = pd.read_csv(training_abstracts_path, sep='\t', header=-1,
        #                 names=abstracts_columns_list)

        # Testing
        testing_abstracts_path = os.getenv('testing_abstracts_path')
        D = pd.read_csv(testing_abstracts_path, sep='\t', header=-1,
                        names=abstracts_columns_list)
        d = np.array([list(D['PMID']), list(D['Title']), list(D['Abstract'])])
        d_array = d.T
        sen_list = []
        for i in range(d_array.shape[0]):
            list1 = []
            list1.append(d_array[i][1])
            for j in d_array[i][2].split('. '):
                list1.append(j)
            sen_list.append(list1)
        # print(sen_list[:2])
        abs_dic = {}
        title_len_dic = {}
        sen_dic = {}
        for i in range(d_array.shape[0]):
            abs_dic[d_array[i][0]] = d_array[i][1] + ' ' + d_array[i][2]
            title_len_dic[d_array[i][0]] = len(d_array[i][1]) + 1
            sen_dic[d_array[i][0]] = sen_list[i]
        # sen_list: [[sentences for paper 1],[sentences for paper 2]]
        # print(abs_dic['10076535'])
        # print(sen_dic['10076535'])
        return abs_dic, sen_dic
    
    def Get_Entities_Dic(self):
        '''Get the dictionary list and gene dictionary.

        Return:
             return a chemical entities list and a gene entities list.
        '''

        entities_columns_list = ['PMID', 'team_num', 'entities_type',
                                 'start_off', 'end_off', 'entity_text']
        # Training
        # training_entities_path = os.getenv('training_entities_path')
        # D = pd.read_csv(training_entities_path, sep='\t', header=-1,
        #                 names=entities_columns_list)

        # Testing
        testing_entities_path = os.getenv('testing_entities_path')
        D = pd.read_csv(testing_entities_path, sep='\t', header=-1,
                        names=entities_columns_list)
        d = np.array([list(D['PMID']), list(D['team_num']),
            list(D['entities_type']), list(D['start_off']),
            list(D['end_off']), list(D['entity_text'])])
        d_array = d.T
        D1 = D.drop_duplicates('PMID')
        PMID_num = list(D1.index)
        PMID_num.append(d_array.shape[0])
        entity_dic = {}
        dic1 = {}
        k = 0
        for j in PMID_num[1:]:
            i = PMID_num[k]
            list1 = []
            list2 = []
            list3 = []
            dic3 = {}
            while i < d_array.shape[0] and i < j:
                offset_list = []
                dic2 = {}
                offset_list.append(int(d_array[i][3]))
                offset_list.append(int(d_array[i][4]))
                dic2['offset'] = offset_list
                dic2['text'] = d_array[i][5]
                dic3[d_array[i][1]] = dic2
                dic1[d_array[i][0]] = dic3
                if d_array[i][2] == 'CHEMICAL':
                    list1.append(d_array[i][1])
                    # dic1['chemical'] = list1

                else:
                    list2.append(d_array[i][1])
                    # dic1['gene'] = list2
                i = i + 1
            list3.append(list1)
            list3.append(list2)
            entity_dic[d_array[PMID_num[k]][0]] = list3
            k = k + 1
        # print(entity_dic['23532918'])
        # print(dic1['23532918'])
        return entity_dic, dic1
    
    def get_all_pair_dic(self, entity_dic):
        '''Get all pairs dictionary.

        Args:
            entity_dic(dict): all entities dictionary.
        Return:
            return all paris dictionary.
        '''
        all_pair_dic = {}
        for i in entity_dic:
            PMID_pair_list = []
            for j in entity_dic[i][0]:
                for k in entity_dic[i][1]:
                    list1 = []
                    list1.append(j)
                    list1.append(k)
                    PMID_pair_list.append(list1)
                    all_pair_dic[i] = PMID_pair_list
        # print(all_pair_dic['23532918'])[['T1', 'T19'], ['T1', 'T20']]]
        return all_pair_dic

    def extract_entities_pairs(self, dic1):
        '''Get the positive pairs

        Args:
             dic(a dictionary): {'(PMID)':{'T1':{'text':'xx', 'offset':[]}}}

        Return:
             return a list containing all positive pairs
        '''
        entities_columns_list = ['PMID', 'CPR_Group', 'Evaluation_type',
                                 'Relation', 'Arg1', 'Arg2']
        # Training
        # training_relations_path = os.getenv('training_relations_path')
        # D = pd.read_csv(training_relations_path, sep='\t', header=-1,
        #                 names=entities_columns_list)
        # Testing
        testing_relations_path = os.getenv('testing_relations_path')
        D = pd.read_csv(testing_relations_path, sep='\t', header=-1,
                        names=entities_columns_list)
        d = np.array([list(D['PMID']), list(D['Arg1']),
                      list(D['Arg2']), list(D['Relation'])])
        d_array = d.T

        D = D.drop_duplicates('PMID')
        PMID_num = list(D.index)
        PMID_num.append(d_array.shape[0])
        pair_dic = {}
        k = 0
        for j in PMID_num[1:]:
            i = PMID_num[k]
            list2 = []
            while i < d_array.shape[0] and i < j:
                list1 = []
                d_array[i][1] = d_array[i][1][5:8].rstrip()
                d_array[i][2] = d_array[i][2][5:8].rstrip()
                list1.append(d_array[i][1])
                list1.append(d_array[i][2])
                list1.append(d_array[i][3])
                list2.append(list1)
                pair_dic[d_array[i][0]] = list2
                i = i + 1
            k = k + 1
        # print(pair_dic['11156594'])
        return pair_dic

    def get_negative_T_pairs(self, all_pair_dic, pair_dic):
        '''Get negative pairs.

        Args:
            all_pair_dic(dict): all the pairs for chemical and gene.
            pair_dic(dict): pisitive pairs.

        Return:
            return the negative pairs dictionary.
        '''
        negative_dic = {}
        no_relation_pair_dic = {}
        for i in pair_dic:
            for j in pair_dic[i]:
                list2 = []
                list1 = j[:2]
                list2.append(list1)
                no_relation_pair_dic[i] = list2
        for i in no_relation_pair_dic:
            list1 = no_relation_pair_dic[i]
            list2 = all_pair_dic[i]
            negative_list = [i for i in list2 if i not in list1]
            negative_dic[i] = negative_list
        # print(len(negative_dic['11156594']))
        return negative_dic


    def all_sen_dic(self, sen_dic, dic1, entity_dic, abs_dic):
        '''Get all sentence dictionary.

        Args:
            sen_dic(dict): sentence dictionary for each PMID.

        Return:
            return all sentences dictionary.
        '''
        all_sen_dic = {}
        for i in sen_dic:
            list1 = []
            for j in sen_dic[i]:
                dic_1 = {}
                dic_1['sentence'] = j
                start = abs_dic[i].index(j)
                dic_1['start_off'] = start
                end = start + len(j) -1
                dic_1['end_off'] = end
                dic3 = {}
                for k in entity_dic[i][0]:
                    if start <= dic1[i][k]['offset'][0] <= end:
                        dic2 = {}
                        offset_list = [h-start for h in dic1[i][k]['offset']]
                        dic2['offset'] = offset_list
                        dic2['text'] = dic1[i][k]['text']
                        dic3[k] = dic2
                dic_1['che'] = dic3
                dic4 = {}
                for k in entity_dic[i][1]:
                    if start <= dic1[i][k]['offset'][0] <= end:
                        dic2 = {}
                        offset_list = [h-start for h in dic1[i][k]['offset']]
                        dic2['offset'] = offset_list
                        dic2['text'] = dic1[i][k]['text']
                        dic4[k] = dic2
                dic_1['gene'] = dic4
                che_list = []
                for k in dic3:
                    che_list.append(k)
                gene_list = []
                for k in dic4:
                    gene_list.append(k)
                dic_1['che_list'] = che_list
                dic_1['gene_list'] = gene_list
                list1.append(dic_1)
            all_sen_dic[i] = list1

        # print(all_sen_dic['23194825'])
        return all_sen_dic

    def get_positive(self, pair_dic, all_sen_dic):
        '''Get positive examples.

        Args:
            pair_dic(dict): positive pairs.
            all_sen_dic(dict): all sentences dic.
        Return:
            return a positive training list.
        '''
        positive_train_list = []
        for i in pair_dic:
            for j in pair_dic[i]:
                for k in range(len(all_sen_dic[i])):
                    list1 = all_sen_dic[i][k]['che_list']
                    list2 = all_sen_dic[i][k]['gene_list']
                    if j[0] in list1 and j[1] in list2:
                        sen = all_sen_dic[i][k]['sentence']
                        pos = k
                        break
                    else:
                        # break
                        sen = 'NO'
                if sen == 'NO':
                    continue
                train_dic = collections.OrderedDict()
                train_dic['PMID'] = i
                train_dic['sentence'] = sen.lower()
                Arg1 = collections.OrderedDict()
                Arg2 = collections.OrderedDict()
                Arg1['type'] = 'Chemical'
                Arg1['text'] = all_sen_dic[i][pos]['che'][j[0]]['text']
                Arg1['offset'] = all_sen_dic[i][pos]['che'][j[0]]['offset']
                Arg2['type'] = 'Gene'
                Arg2['text'] = all_sen_dic[i][pos]['gene'][j[1]]['text']
                Arg2['offset'] = all_sen_dic[i][pos]['gene'][j[1]]['offset']
                train_dic['Arg1'] = Arg1
                train_dic['Arg2'] = Arg2
                chen = []
                for k in all_sen_dic[i][pos]['che_list']:
                    if k != j[0]:
                        chen.append(all_sen_dic[i][pos]['che'][k]['offset'])
                genen = []
                for k in all_sen_dic[i][pos]['gene_list']:
                    if k != j[1]:
                        genen.append(all_sen_dic[i][pos]['gene'][k]['offset'])
                train_dic['chen_off_list'] = chen
                train_dic['genen_off_list'] = genen
                train_dic['relation'] = j[2]
                train_dic['raw_sentence'] = sen
                positive_train_list.append(train_dic)
        return positive_train_list

    def get_negative(self, negative_dic, all_sen_dic):
        '''Get negative examples.

        Args:
            negative_dic(dict): negative pair dict.
            all_sen_dic(dcit): all sentences dict.

        Return:
            return a negative training list.
        '''
        negative_train_list = []
        for i in negative_dic:
            for j in negative_dic[i]:
                for k in range(len(all_sen_dic[i])):
                    list1 = all_sen_dic[i][k]['che_list']
                    list2 = all_sen_dic[i][k]['gene_list']
                    if j[0] in list1 and j[1] in list2:
                        sen = all_sen_dic[i][k]['sentence']
                        pos = k
                        break
                    else:
                        sen = 'NO' 
                if sen == 'NO':
                    continue
                train_dic = collections.OrderedDict()
                train_dic['PMID'] = i
                train_dic['sentence'] = sen.lower()
                Arg1 = collections.OrderedDict()
                Arg2 = collections.OrderedDict()
                Arg1['type'] = 'Chemical'
                Arg1['text'] = all_sen_dic[i][pos]['che'][j[0]]['text']
                Arg1['offset'] = all_sen_dic[i][pos]['che'][j[0]]['offset']
                Arg2['type'] = 'Gene'
                Arg2['text'] = all_sen_dic[i][pos]['gene'][j[1]]['text']
                Arg2['offset'] = all_sen_dic[i][pos]['gene'][j[1]]['offset']
                train_dic['Arg1'] = Arg1
                train_dic['Arg2'] = Arg2
                chen = []
                for k in all_sen_dic[i][pos]['che_list']:
                    if k != j[0]:
                        chen.append(all_sen_dic[i][pos]['che'][k]['offset'])
                genen = []
                for k in all_sen_dic[i][pos]['gene_list']:
                    if k != j[1]:
                        genen.append(all_sen_dic[i][pos]['gene'][k]['offset'])
                train_dic['chen_off_list'] = chen
                train_dic['genen_off_list'] = genen
                train_dic['relation'] = 'NEGATIVE'
                train_dic['raw_sentence'] = sen
                negative_train_list.append(train_dic)
        print('len of negative train list', len(negative_train_list))
        # print(negative_train_list[:10])
        return negative_train_list

    def Get_training_examples(self, positive_train_list, negative_train_list):
        '''Get training examples containing positive and negative

        Args:
             positive_train_list(a list): positive examples
             negative_train_list(a list): negative examples
        '''
        training_examples_list = positive_train_list
        negative_examples = random.sample(negative_train_list, 6437)
        training_examples_list.extend(negative_examples)
        print('Get the training examples')
        with open('./chemprot_training/training_examples.json', 'w+') as f:
            json.dump(training_examples_list, f, indent=4)
        print('Get the training examples done')

    def get_testing_examples(self, positive_train_list):
        '''
        '''
        training_examples = positive_train_list
        with open('./chemprot_test_gs/testing_examples.json', 'w+') as f:
            json.dump(training_examples, f, indent=4)

    def replace_entities(self):
        '''Replace entities with tokens: CHEMICAL, GENE, CHENICALN and GENEN.
        '''
        # with open('./chemprot_training/training_examples.json', 'r') as f:
        with open('./chemprot_test_gs/testing_examples.json', 'r') as f:
            training_examples = json.load(f)
            new_training_examples = []
            print(len(training_examples))
            for i in training_examples:
                tmp = i['sentence']
                text = i['sentence']
                all_off_list = []
                che_off = i['Arg1']['offset']
                che_off.append(' CHEMICAL1 ')
                # print(che_off)
                all_off_list.append(che_off)
                gene_off = i['Arg2']['offset']
                gene_off.append(' GENE1 ')
                # print(gene_off)
                all_off_list.append(gene_off)
                chen_off = i['chen_off_list']
                genen_off = i['genen_off_list']
                
                chen_text_list = []
                for j in chen_off:
                    if len(j):
                        chen_text_list.append(tmp[j[0]:j[1]].lower())
                genen_text_list = []
                for j in genen_off:
                    if len(j):
                        genen_text_list.append(tmp[j[0]:j[1]].lower())

                for j in chen_off:
                    if len(j):
                        if text[j[0]:j[1]] == i['Arg1']['text'].lower():
                            j.append(' CHEMICAL1 ')
                            all_off_list.append(j)
                for j in genen_off:
                    if len(j):
                        if text[j[0]:j[1]] == i['Arg2']['text'].lower():
                            j.append(' GENE1 ')
                            all_off_list.append(j)
                all_off_list.sort()
                list1 = all_off_list[::-1]
                # print(list1)
                for j in list1:
                    text = text[:j[0]] + j[2] + text[j[1]:]
                for j in chen_text_list:
                    if ' ' + j + ' ' in text:
                        text = text.replace(j, ' CHEMICALN ')
                for j in genen_text_list:
                    if ' ' + j + ' ' in text:
                        text = text.replace(j, ' GENEN ')
                i['sentence'] = text
                text = text + ' '
                # if 'GENE ' not in text or 'CHEMICAL ' not in text:
                #     training_examples.remove(i)
                if 'GENE1 ' in text and 'CHEMICAL1 ' in text:
                    new_training_examples.append(i)
            # new_training_examples = training_examples
            print(len(new_training_examples))
        # with open('./chemprot_training/new_training_examples.json', 'w+') as j:
        with open('./chemprot_test_gs/new_testing_examples.json', 'w+') as j:
            json.dump(new_training_examples, j, indent=4)

    def genia_tokenizer(self):
        '''Tokenize pair text with genia tagger.

        '''
        tagger = GeniaTagger('./tools/geniatagger-3.0.2/geniatagger')
        with open('./chemprot_test_gs/new_testing_examples.json', 'r') as f:
            training_examples = json.load(f)
            # print(len(training_examples))
            for i in training_examples:
                tokenized_tuple = tagger.parse(i['sentence'])
                token_list = []
                for output in tokenized_tuple:
                    if output[0] in string.punctuation:
                        continue
                    pos = output[2]
                    if output[0] == pos:
                        continue
                    if output[0].endswith('..'):
                        token = output[0][:-2]
                    elif pos == 'CD':
                        token = 'NUM'
                    else:
                        token = output[0]
                    token_list.append(token)
                i['sentence'] = ' '.join(token_list)
        # with open('./chemprot_training/train_tokenized.json', 'w+') as j:
        with open('./chemprot_test_gs/testing_tokenized.json', 'w+') as j:
            json.dump(training_examples, j, indent=4)
    
    def remove_stop_words(self):
        '''Remove stop words from the tokenized text of pairs.

        '''
        with open ('./other/stop-word-list.txt') as f:
            stop_word_list = [word.strip() for word in f]
        # with open('./chemprot_training/train_tokenized.json', 'r') as j:
        with open('./chemprot_test_gs/testing_tokenized.json', 'r') as j:
            training_examples = json.load(j)
            # print(len(training_examples))
            for i in training_examples:
                sentence = i['sentence']
                token_list = sentence.split()
                new_token_list = []
                for token in token_list:
                    if token not in stop_word_list:
                        new_token_list.append(token)
                i['sentence'] = ' '.join(new_token_list)
        # with open('./chemprot_training/remove_stop_word.json', 'w+') as k:
        with open('./chemprot_test_gs/remove_stop_word.json', 'w+') as k:
            json.dump(training_examples, k, indent=4)

    def generate_entitiesN_pos(self):
        '''Generate token positions by the CHEMICAL and GENE

        '''
        def shrink(offset):
            '''Shrink the offset range.

            Args:
                offset (int): The shrinking offset.
            '''
            if -5 <= offset <= 5:
                return offset
            elif 6 <= abs(offset) <=10:
                return -6 if offset < 0 else 6
            elif 11 <= abs(offset) <=20:
                return -7 if offset < 0 else 7
            elif 21 <= abs(offset) <= 30:
                return -8 if offset < 0 else 8
            else:
                return -9 if offset < 0 else 9

        def find_all_index(arr,item):
            return [i for i,a in enumerate(arr) if a==item]

        # with open('./chemprot_training/train_dataset.json', 'r+') as f:
        with open('./chemprot_test_gs/remove_stop_word.json', 'r+') as f:
            training_examples = json.load(f)
            for i in training_examples:
                token_list = i['sentence'].split()
                len_sen = len(token_list)
                e1_index = token_list.index('CHEMICAL1')
                e2_index = token_list.index('GENE1')
                e1_pos_list = [shrink(j-e1_index) for j in range(len_sen)]
                same_chemical_list = find_all_index(token_list, 'CHEMICAL1')
                for j in same_chemical_list:
                    e1_pos_list[j] = 0
                assert len(e1_pos_list) == len_sen
                e2_pos_list = [shrink(j-e2_index) for j in range(len_sen)]
                same_gene_list = find_all_index(token_list, 'GENE1')
                for j in same_gene_list:
                    e2_pos_list[j] = 0
                assert len(e2_pos_list) == len_sen
                i['che_pos'] = e1_pos_list
                i['gene_pos'] = e2_pos_list
        # a = 0
        # for i in training_examples:
        #     if i['relation'] != 'NEGATIVE':
        #         a = a + 1
        # print(a)
        # with open('./chemprot_training/train_dataset.json', 'w+') as j:
        with open('./chemprot_test_gs/test_dataset.json', 'w+') as j:
            json.dump(training_examples, j, indent=4)
        

    def match_emb(self):
        '''Match the tokens to the pre-trained word embedding vocabulary.

        '''
        vocab = self.pretrained_emb.vocab

        fix_dict_path = os.getenv('fix_dict_path')
        with open(fix_dict_path) as f:
            fix_dict = json.load(f)

        # with open('./chemprot_training/train_dataset.json', 'r+') as f:
        with open('./chemprot_test_gs/test_dataset.json', 'r+') as f:
            training_examples = json.load(f)
            unmatched_set = set()
            for i in training_examples:
                a = i['sentence'].replace('-', ' ')
                token_list = a.split()
                matched_token_list = []
                for token in token_list:
                    if token in fix_dict:
                        token = fix_dict[token]
                    if token in self.extra_token_list:
                        pass
                    elif token not in vocab:
                        if token.capitalize() in vocab:
                            token = token.capitalize()
                        elif token.upper() in vocab:
                            token = token.upper()
                        else:
                            unmatched_set.add(token)
                    matched_token_list.append(token)
                i['sentence'] = ' '.join(matched_token_list)
        # Training
        # with open('./chemprot_training/train_dataset.json', 'w+') as j:
        # Testing
        # with open('./chemprot_test_gs/test_dataset.json', 'w+') as j:
            # json.dump(training_examples, j, indent=4)
            print(unmatched_set)
            print(len(unmatched_set))
    
    def get_all_data_list(self):
        '''Get train list.

        Return:
            return the train list
        '''
        with open('./chemprot_training/train_dataset.json', 'r+') as f:
            train_list = json.load(f)
        with open('./chemprot_test_gs/test_dataset.json', 'r+') as j:
            test_list = json.load(j)
        return train_list, test_list
    
    def construct_emb(self):
        '''Construct the word embedding tensor and vocabulary list.
        '''
        emb_vocab_path = os.getenv('emb_vocab_path')
        emb_vec_path = os.getenv('emb_vec_path')
        train_vocab = {'': 0}
        emb_dim = self.pretrained_emb.vector_size
        train_vec_list = [[[0] * emb_dim]]
        vocab = self.pretrained_emb.vocab

        pair_list = self.train_list# + self.test_list
        with progressbar.ProgressBar(max_value=len(pair_list)) as bar:
            for i, pair in enumerate(pair_list):
                token_list = pair['sentence'].strip().split()
                for token in token_list:
                    if token not in train_vocab:
                        if token in vocab:
                            train_vocab[token] = len(train_vocab)
                            train_vec_list.append([self.pretrained_emb[token]])
                        elif token not in self.extra_token_list:
                            raise ValueError('UNKNOWN TOKEN ' + token)
                bar.update(i)
        emb_mat = np.concatenate(train_vec_list)

        for extra_token in self.extra_token_list:
            train_vocab[extra_token] = len(train_vocab)
            extra_tensor = np.random.randn(1, emb_dim)
            emb_mat = np.concatenate((emb_mat, extra_tensor))

        print('VALIDATING.')
        for token, index in train_vocab.items():
            if (token not in self.extra_token_list) and token in vocab:
                old_vec = self.pretrained_emb[token]
                new_vec = emb_mat[index]
                try:
                    assert np.array_equal(old_vec, new_vec)
                except AssertionError:
                    print('FAILED ' + token)
        print('VALIDATION FINISHED.')
        
        with open(emb_vocab_path, 'w+') as f:
            json.dump(train_vocab, f, indent=4)
        self.train_vocab = train_vocab
        np.save(emb_vec_path, emb_mat)

    def token2ix(self):
        '''Convert tokens to the list of vocabulary indices.
        '''
        vocab = self.train_vocab
        # with open('./chemprot_training/train_dataset.json', 'r+') as f:
        with open('./chemprot_test_gs/test_dataset.json', 'r+') as f:
            training_examples = json.load(f)
            for i in training_examples:
                token_list = i['sentence'].split()
                i['index_list'] = [vocab[token] for token in token_list]
        # with open('./chemprot_training/train_dataset.json', 'w+') as j:
        with open('./chemprot_test_gs/test_dataset.json', 'w+') as j:
            json.dump(training_examples, j, indent=4)

if __name__ == '__main__':
    a = Preprocessor()
    # abs_dic, sen_dic = a.Extract_Sentences()
    # entity_dic, dic1 = a.Get_Entities_Dic()
    # pair_dic = a.extract_entities_pairs(dic1)
    # all_pair_dic = a.get_all_pair_dic(entity_dic)
    # negative_dic = a.get_negative_T_pairs(all_pair_dic, pair_dic)
    # all_sen_dic = a.all_sen_dic(sen_dic, dic1, entity_dic, abs_dic)
    # positive_train_list = a.get_positive(pair_dic, all_sen_dic)
    # negative_train_list = a.get_negative(negative_dic, all_sen_dic)
    # a.Get_training_examples(positive_train_list, negative_train_list)
    # a.get_testing_examples(positive_train_list)
    # a.replace_entities()
    # a.genia_tokenizer()
    # a.remove_stop_words()
    a.load_pretrained_emb()
    a.match_emb()
    # a.generate_entitiesN_pos()
    # a.get_all_data_list()
    # a.construct_emb()
    # a.token2ix()
