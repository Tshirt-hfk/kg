import json
import os
from random import choice

import numpy as np
import pyhanlp
from gensim.models import KeyedVectors


# 分词
def tokenize(s):
    return [i.word for i in pyhanlp.HanLP.segment(s)]


# 句子padding
def seq_padding(X, padding=0):
    """
    :param X
    :param padding:
    :return: (bsz, len)
    """
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


# 句子的词转id
def sent2id(S, word2id):
    """
    S格式：[[w1, w2]]
    return (bsz, len)
    """
    V = []
    for s in S:
        V.append([])
        for w in s:
            for _ in w:
                V[-1].append(word2id.get(w, 0))
    V = seq_padding(V)
    return V


def load_word_vec(wordVecPath="./word2vec_baike/sgns.baidubaike.bigram-char", padding_idx=0):
    # 词向量模型加载
    print("start word2vec load ......")
    wv_from_text = KeyedVectors.load_word2vec_format(wordVecPath, binary=False,
                                                     encoding="utf8", unicode_errors='ignore')
    print("word2vec load succeed")

    id2word = {i + 1: j for i, j in enumerate(wv_from_text.index2word)}
    word2id = {j: i for i, j in id2word.items()}
    word_vec = wv_from_text.vectors
    word_vec = np.concatenate([np.zeros((1, word_vec.shape[1])), word_vec])
    return word2id, word_vec


def load_data(mode=0):
    # 数据加载
    total_data = json.load(open('datasets/train_data_me.json', encoding='utf-8'))
    id2predicate, predicate2id = json.load(open('datasets/all_schemas_me.json', encoding='utf-8'))
    id2predicate = {int(i): j for i, j in id2predicate.items()}
    id2char, char2id = json.load(open('datasets/all_chars_me.json', encoding='utf-8'))
    num_classes = len(id2predicate)
    char_num = len(char2id) + 2  # padding and mask

    # 打乱后的训练数据id
    if not os.path.exists('random_order_vote.json'):
        random_order = np.arange(len(total_data))
        np.random.shuffle(random_order)
        json.dump(
            random_order.tolist(),
            open('random_order_vote.json', 'w', encoding='utf-8'),
            indent=4
        )
    else:
        random_order = json.load(open('random_order_vote.json', encoding='utf-8'))

    # 数据分割
    train_data = [total_data[j] for i, j in enumerate(random_order) if i % 8 != mode]
    dev_data = [total_data[j] for i, j in enumerate(random_order) if i % 8 == mode]

    predicates = {}  # 格式：{predicate: [(subject, predicate, object)]}

    for d in train_data:
        # repair(d)
        for sp in d['spo_list']:
            if sp[1] not in predicates:
                predicates[sp[1]] = []
            predicates[sp[1]].append(sp)

    return train_data, dev_data, char_num, num_classes, predicates, id2char, char2id, id2predicate, predicate2id


def random_generate(d, predicates):
    """
    随机替换同relation的subject、object
    :param d:
    :param spo_list_key:
    :param predicates:  关系list
    :return:
    """
    r = np.random.random()
    if r > 0.5:
        return d
    else:
        k = np.random.randint(len(d['spo_list']))
        spi = d['spo_list'][k]
        k = np.random.randint(len(predicates[spi[1]]))
        spo = predicates[spi[1]][k]
        F = lambda s: s.replace(spi[0], spo[0]).replace(spi[2], spo[2])
        text = F(d['text'])
        spo_list = [(F(sp[0]), sp[1], F(sp[2])) for sp in d['spo_list']]
        return {'text': text, 'spo_list': spo_list}


class data_generator:
    def __init__(self, data, char2id, word2id, predicate2id, num_classes, predicates, batch_size=64, maxlen=512):
        self.data = data
        self.char2id = char2id
        self.word2id = word2id
        self.predicate2id = predicate2id
        self.num_classes = num_classes
        self.predicates = predicates
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        idxs = np.arange(len(self.data))
        np.random.shuffle(idxs)
        T1, T2, S1, S2, K, O1, O2, = [], [], [], [], [], [], []
        for i in idxs:
            d = random_generate(self.data[i], self.predicates)
            text = d['text'][:self.maxlen]
            text_words = tokenize(text)
            text = ''.join(text_words)
            items = {}
            for sp in d['spo_list']:
                subjectid = text.find(sp[0])
                objectid = text.find(sp[2])
                if subjectid != -1 and objectid != -1:
                    key = (subjectid, subjectid + len(sp[0]))
                    if key not in items:
                        items[key] = []
                    items[key].append((objectid,
                                       objectid + len(sp[2]),
                                       self.predicate2id[sp[1]]))
            if items:
                T1.append([self.char2id.get(c, 1) for c in text])  # 1是unk，0是padding
                T2.append(text_words)
                s1, s2 = np.zeros(len(text)), np.zeros(len(text))
                for j in items:
                    s1[j[0]] = 1
                    s2[j[1] - 1] = 1

                k1, k2 = choice(list(items.keys()))
                k = np.zeros(len(text))
                for j in range(k1, k2):
                    k[j] = 1

                o1, o2 = np.zeros((len(text), self.num_classes)), np.zeros((len(text), self.num_classes))
                for j in items.get((k1, k2), []):
                    o1[j[0]][j[2]] = 1
                    o2[j[1] - 1][j[2]] = 1
                S1.append(s1)
                S2.append(s2)
                K.append(k)
                O1.append(o1)
                O2.append(o2)
                if len(T1) == self.batch_size or i == idxs[-1]:
                    T1 = seq_padding(T1).astype('int64')
                    T2 = sent2id(T2, self.word2id).astype('int64')
                    S1 = seq_padding(S1).astype('int64')
                    S2 = seq_padding(S2).astype('int64')
                    K = seq_padding(K).astype('float32')
                    O1 = seq_padding(O1, np.zeros(self.num_classes)).astype('int64')
                    O2 = seq_padding(O2, np.zeros(self.num_classes)).astype('int64')
                    yield [T1, T2, S1, S2, K, O1, O2]
                    T1, T2, S1, S2, K, O1, O2, = [], [], [], [], [], [], []
        return
