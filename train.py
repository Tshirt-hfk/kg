from __future__ import print_function

import json
import os
from random import choice

import pyhanlp

# 词向量模型加载
from paddle.fluid.dygraph import to_variable
from tqdm import tqdm

from KGModel import *

print("start word2vec load ......")
from gensim.models import KeyedVectors

wv_from_text = KeyedVectors.load_word2vec_format('./word2vec_baike/sgns.baidubaike.bigram-char',
                                                 binary=False, encoding="utf8",
                                                 unicode_errors='ignore')  # C text format
print("word2vec load succeed")

id2word = {i + 1: j for i, j in enumerate(wv_from_text.wv.index2word)}
word2id = {j: i for i, j in id2word.items()}
word2vec = wv_from_text.wv.syn0
word_dim = word2vec.shape[1]
word2vec = np.concatenate([np.zeros((1, word_dim)), word2vec])

mode = 0
embedding_dim = 128
maxlen = 512
padding_idx = 0


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


# 句子的词转向量
def sent2vec(S):
    """
    S格式：[[w1, w2]]
    return (bsz, len, embedding_dim)
    """
    V = []
    for s in S:
        V.append([])
        for w in s:
            for _ in w:
                V[-1].append(word2id.get(w, 0))
    V = seq_padding(V)
    V = word2vec[V]
    return V


# 数据加载
total_data = json.load(open('datasets/train_data_me.json', encoding='utf-8'))
id2predicate, predicate2id = json.load(open('datasets/all_schemas_me.json', encoding='utf-8'))
id2predicate = {int(i): j for i, j in id2predicate.items()}
id2char, char2id = json.load(open('datasets/all_chars_me.json', encoding='utf-8'))
num_classes = len(id2predicate)

char_num = len(char2id) + 2

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

# def repair(d):
#     d['text'] = d['text'].lower()
#     something = re.findall(u'《([^《》]*?)》', d['text'])
#     something = [s.strip() for s in something]
#     zhuanji = []
#     gequ = []
#     for sp in d['spo_list']:
#         sp[0] = sp[0].strip(u'《》').strip().lower()
#         sp[2] = sp[2].strip(u'《》').strip().lower()
#         for some in something:
#             if sp[0] in some and d['text'].count(sp[0]) == 1:
#                 sp[0] = some
#         if sp[1] == u'所属专辑':
#             zhuanji.append(sp[2])
#             gequ.append(sp[0])
#     spo_list = []
#     for sp in d['spo_list']:
#         if sp[1] in [u'歌手', u'作词', u'作曲']:
#             if sp[0] in zhuanji and sp[0] not in gequ:
#                 continue
#         spo_list.append(tuple(sp))
#     d['spo_list'] = spo_list


for d in train_data:
    # repair(d)
    for sp in d['spo_list']:
        if sp[1] not in predicates:
            predicates[sp[1]] = []
        predicates[sp[1]].append(sp)


# for d in dev_data:
#     repair(d)


def random_generate(d, spo_list_key):
    """
    随机替换同relation的subject、object
    :param d:
    :param spo_list_key:
    :return:
    """
    r = np.random.random()
    if r > 0.5:
        return d
    else:
        k = np.random.randint(len(d[spo_list_key]))
        spi = d[spo_list_key][k]
        k = np.random.randint(len(predicates[spi[1]]))
        spo = predicates[spi[1]][k]
        F = lambda s: s.replace(spi[0], spo[0]).replace(spi[2], spo[2])
        text = F(d['text'])
        spo_list = [(F(sp[0]), sp[1], F(sp[2])) for sp in d[spo_list_key]]
        return {'text': text, spo_list_key: spo_list}


class data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
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
            spo_list_key = 'spo_list'  # if np.random.random() > 0.5 else 'spo_list_with_pred'
            d = random_generate(self.data[i], spo_list_key)
            text = d['text'][:maxlen]
            text_words = tokenize(text)
            text = ''.join(text_words)
            items = {}
            for sp in d[spo_list_key]:
                subjectid = text.find(sp[0])
                objectid = text.find(sp[2])
                if subjectid != -1 and objectid != -1:
                    key = (subjectid, subjectid + len(sp[0]))
                    if key not in items:
                        items[key] = []
                    items[key].append((objectid,
                                       objectid + len(sp[2]),
                                       predicate2id[sp[1]]))
            if items:
                T1.append([char2id.get(c, 1) for c in text])  # 1是unk，0是padding
                T2.append(text_words)
                s1, s2 = np.zeros(len(text)), np.zeros(len(text))
                for j in items:
                    s1[j[0]] = 1
                    s2[j[1] - 1] = 1

                k1, k2 = choice(list(items.keys()))
                k = np.zeros(len(text))
                for j in range(k1, k2):
                    k[j] = 1

                # k1, k2 = np.array(list(items.keys())).T
                # k1 = choice(k1)
                # k2 = choice(k2[k2 >= k1])

                o1, o2 = np.zeros((len(text), num_classes)), np.zeros((len(text), num_classes))
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
                    T2 = sent2vec(T2).astype('float32')
                    S1 = seq_padding(S1).astype('int64')
                    S2 = seq_padding(S2).astype('int64')
                    O1 = seq_padding(O1, np.zeros(num_classes)).astype('int64')
                    O2 = seq_padding(O2, np.zeros(num_classes)).astype('int64')
                    K = seq_padding(K).astype('float32')
                    yield [T1, T2, S1, S2, K, O1, O2]
                    T1, T2, S1, S2, K, O1, O2, = [], [], [], [], [], [], []
        return


# t1 = layers.data(name="t1", shape=(None, None), dtype="int64")
# t2 = layers.data(name="t2", shape=(None, None, word_dim), dtype="float32")
# s1 = layers.data(name="s1", shape=(None, None), dtype="int64")
# s2 = layers.data(name="s2", shape=(None, None), dtype="int64")
# k1 = layers.data(name="k1", shape=(None, 1), dtype="int32")
# k2 = layers.data(name="k2", shape=(None, 1), dtype="int32")
# o1 = layers.data(name="o1", shape=(None, num_classes), dtype="int64")
# o2 = layers.data(name="o2", shape=(None, num_classes), dtype="int64")


def extract_items(model, text_in):
    text_words = tokenize(text_in.lower())
    text_in = ''.join(text_words)
    R = []
    _t1 = [char2id.get(c, 1) for c in text_in]
    _t1 = np.array([_t1])
    _t2 = sent2vec([text_words])

    t1 = to_variable(_t1.astype("int64"))
    t2 = to_variable(_t2.astype("float32"))
    mask = get_mask(_t1)
    t = model.embedding(t1, t2)
    t, pn1, pn2, ps1, ps2 = model.er_model(t, mask=mask)
    _ps1, _ps2 = ps1.numpy(), ps2.numpy()

    _k1, _k2 = _ps1[0, :, 0], _ps2[0, :, 0]
    _k1, _k2 = np.where(_k1 > 0.5)[0], np.where(_k2 > 0.4)[0]
    _subjects = []
    for i in _k1:
        j = _k2[_k2 >= i]
        if len(j) > 0:
            j = j[0]
            _subject = text_in[i: j + 1]
            k = np.zeros(len(text_in))
            for x in range(i, j):
                k[j] = x
            _subjects.append((_subject, k))
    if _subjects:
        _t1 = np.repeat(_t1, len(_subjects), 0)
        _t2 = np.repeat(_t2, len(_subjects), 0)
        _k = np.array([_s[1] for _s in _subjects])

        k = to_variable(_k.astype("float32"))
        _o1, _o2 = model.re_model(t, k, pn1, pn2, mask=mask)

        for i, _subject in enumerate(_subjects):
            _oo1, _oo2 = np.where(_o1[i] > 0.5), np.where(_o2[i] > 0.4)
            for _ooo1, _c1 in zip(*_oo1):
                for _ooo2, _c2 in zip(*_oo2):
                    if _ooo1 <= _ooo2 and _c1 == _c2:
                        _object = text_in[_ooo1: _ooo2 + 1]
                        _predicate = id2predicate[_c1]
                        R.append((_subject[0], _predicate, _object))
                        break
        zhuanji, gequ = [], []
        for s, p, o in R[:]:
            if p == u'妻子':
                R.append((o, u'丈夫', s))
            elif p == u'丈夫':
                R.append((o, u'妻子', s))
            if p == u'所属专辑':
                zhuanji.append(o)
                gequ.append(s)
        spo_list = set()
        for s, p, o in R:
            if p in [u'歌手', u'作词', u'作曲']:
                if s in zhuanji and s not in gequ:
                    continue
            spo_list.add((s, p, o))
        return list(spo_list)
    else:
        return []


def evaluate(model):
    orders = ['subject', 'predicate', 'object']
    A, B, C = 1e-10, 1e-10, 1e-10
    F = open('dev_pred.json', 'w')
    for d in tqdm(iter(dev_data)):
        R = set([json.dumps(item) for item in extract_items(model, d['text'])])
        T = set([json.dumps(item) for item in d['spo_list']])
        A += len(R & T)
        B += len(R)
        C += len(T)
        s = json.dumps({
            'text': d['text'],
            'spo_list': [
                dict(zip(orders, json.loads(spo))) for spo in T
            ],
            'spo_list_pred': [
                dict(zip(orders, json.loads(spo))) for spo in R
            ],
            'new': [
                dict(zip(orders, json.loads(spo))) for spo in R - T
            ],
            'lack': [
                dict(zip(orders, json.loads(spo))) for spo in T - R
            ]
        }, ensure_ascii=False, indent=4)
        F.write(s + '\n')
    F.close()
    return 2 * A / (B + C), A / B, A / C


if __name__ == "__main__":
    EPOCH_NUM = 120

    use_gpu = True
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        kg_model = KGModel(embedding_dim, word_dim, maxlen, char_num, num_classes, padding_idx)

        adam = fluid.optimizer.AdamOptimizer(learning_rate=layers.linear_lr_warmup(learning_rate=1e-3,
                                                                                   warmup_steps=90,
                                                                                   start_lr=1e-6,
                                                                                   end_lr=1e-3),
                                             parameter_list=kg_model.parameters())

        # global_steps = layers.learning_rate_scheduler._decay_step_counter()
        #
        # ema = fluid.optimizer.ExponentialMovingAverage(0.9999, thres_steps=global_steps)
        #
        # ema.update()

        for epoch in range(EPOCH_NUM):
            kg_model.train()
            for step, data in enumerate(data_generator(train_data)):
                t1, t2, s1, s2, k, o1, o2 = data
                t1, t2 = to_variable(t1), to_variable(t2)
                s1, s2 = to_variable(s1), to_variable(s2)
                k = to_variable(k)
                o1, o2 = to_variable(o1), to_variable(o2)

                ps1, ps2, po1, po2, mask = kg_model(t1, t2, k)

                loss = loss_func(ps1, ps2, s1, s2, po1, po2, o1, o2, mask)

                loss.backward()

                adam.minimize(loss)

                kg_model.clear_gradients()

                print("epoch {} step {}: ".format(epoch, step), loss.numpy()[0])

            kg_model.eval()
            result = evaluate(kg_model)
            print(result)
