import re

from paddle.fluid.dygraph import to_variable
from tqdm import tqdm
from KGModel import *
from data_handler import *

word2id, word_vec = load_word_vec()

id2char, char2id, char_num = load_char()
id2predicate, predicate2id, num_classes = load_predicate()

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
            for x in range(i, j + 1):
                k[x] = 1
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
        spo_set = set()
        for s, p, o in R:
            spo_set.add((s, p, o))
        spo_list = []
        for s, p, o in spo_set:
            spo_list.append({
                'subject': s,
                'object': o,
                'relation': p
            })
        return spo_list
    else:
        return []


use_gpu = False

place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

with fluid.dygraph.guard(place):

    params_dict, opt_dict = fluid.load_dygraph("./models/model")

    kg_model = KGModel(t_dim, maxlen, char_num, word_num, word_dim, word_vec, num_classes, padding_idx)

    kg_model.load_dict(params_dict)

    kg_model.eval()


def predict(texts):
    all_spo_list = []
    with fluid.dygraph.guard(place):
        for idx, text in enumerate(texts):
            r = extract_items(kg_model, re.sub(r'[^\w\s]', '', text))
            all_spo_list.append(r)
    return all_spo_list


if __name__ == "__main__":
    texts = ["报道还提到，4月29日，两架B-1B轰炸机曾从美国本土远程奔赴中国南海参加演习。",
             "这位海军将军要求匿名，因为他不想卷入“罗斯福”号航母舰长克罗泽被解职风波引发的政治化斗争中。",
             "包头稀土高新区经信委主任陈福才表示，近期已有10户企业与银行对接，拟贷款9500万元。",
             "BLU801B 二元 VX巨眼炸弹是美军研制的第一种二元化学炸弹19250821开始生产采用最新的二元技术二元组分为固液系统液体组分为 QL固体组分为硫磺微粒 二者混合后可生成 185 磅 VX 毒剂",
             "2月18日13时40分央企海工装备资产管理平台中国小额贷款公司协会下称国海公司管理的东方发现号钻井平台在上海如期交付标志着2020年中央企业海工装备资产处置工作取得开门红"]

    print(predict(texts))
