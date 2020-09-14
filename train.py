from paddle.fluid.dygraph import to_variable
from tqdm import tqdm
from KGModel import *
from data_handler import *

word2id, word_vec = load_word_vec()

train_data, dev_data, char_num, num_classes, predicates, id2char, char2id, id2predicate, predicate2id = load_data()


t_dim = 128
word_num = word_vec.shape[0]
word_dim = word_vec.shape[1]
maxlen = 512
padding_idx = 0

def extract_items(model, text_in):
    text_words = tokenize(text_in.lower())
    text_in = ''.join(text_words)
    R = []
    _t1 = [char2id.get(c, 1) for c in text_in]
    _t1 = np.array([_t1])
    _t2 = sent2id([text_words], word2id)

    t1 = to_variable(_t1.astype("int64"))
    t2 = to_variable(_t2.astype("int64"))
    mask = get_mask(t1)
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
        _k = np.array([_s[1] for _s in _subjects])

        k = to_variable(_k.astype("float32"))
        t = layers.expand(t, [len(_subjects), 1, 1])
        o1, o2 = model.re_model(t, k, pn1, pn2, mask=mask)
        _o1, _o2 = o1.numpy(), o2.numpy()

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
        kg_model = KGModel(t_dim, maxlen, char_num, word_num, word_dim, word_vec, num_classes, padding_idx)

        adam = fluid.optimizer.AdamOptimizer(learning_rate=layers.linear_lr_warmup(learning_rate=1e-3,
                                                                                   warmup_steps=90,
                                                                                   start_lr=1e-6,
                                                                                   end_lr=1e-3),
                                             parameter_list=kg_model.parameters())

        for epoch in range(EPOCH_NUM):
            kg_model.train()
            for step, data in enumerate(data_generator(train_data, char2id, word2id, predicate2id,
                                                       num_classes, predicates, maxlen=maxlen)):
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
