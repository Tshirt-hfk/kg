import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.dygraph import nn


def seq_gather(seq, idxs):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，
    在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    idxs = layers.cast(idxs, dtype="int32")
    batch_idxs = layers.arange(0, seq.shape[0], dtype="int32")
    batch_idxs = layers.unsqueeze(batch_idxs, 1)
    idxs = layers.concat([batch_idxs, idxs], 1)
    return layers.gather_nd(seq, idxs)


def get_k_inter(seq, k):
    k = layers.unsqueeze(k, -1)
    seq_mean = seq * k
    seq_mean = layers.reduce_sum(seq_mean, dim=1, keep_dim=True) / layers.reduce_sum(k, dim=1, keep_dim=True)
    seq_max = seq - (1 - k) * 1e10
    seq_max = layers.reduce_max(seq_max, dim=1, keep_dim=True)
    return layers.concat([seq_mean, seq_max], axis=-1)


def position_id(x, r=0):
    pid = layers.arange(0, x.shape[1], dtype="int32")
    pid = layers.unsqueeze(pid, 0)
    r = layers.cast(layers.ones_like(x), dtype="int32") * r
    return layers.cast(layers.abs(layers.elementwise_sub(pid, r)), dtype='int64')


def get_mask(seq, padding_idx=0):
    pix = layers.unsqueeze(layers.ones_like(seq) * padding_idx, axes=2)
    mask = layers.cast(layers.greater_than(layers.unsqueeze(seq, axes=2), pix), 'float32')
    return mask


class Conv1d(fluid.dygraph.Layer):

    def __init__(self, num_channels, num_filters, filter_size, stride=1, padding=0, dilation=1):
        super(Conv1d, self).__init__()
        self.conv2d = nn.Conv2D(num_channels, num_filters, (filter_size, 1),
                                stride=stride, padding=[padding, 0], dilation=[dilation, 1])

    def forward(self, seq):
        seq = layers.transpose(seq, [0, 2, 1])
        seq = layers.unsqueeze(seq, -1)
        seq = self.conv2d(seq)
        seq = layers.squeeze(seq, [-1])
        seq = layers.transpose(seq, [0, 2, 1])
        return seq


class DilatedGatedConv1d(fluid.dygraph.Layer):

    def __init__(self, num_channels, num_filters, filter_size, stride=1, padding=0, dilation=1, dropout_rate=0.1):
        super(DilatedGatedConv1d, self).__init__()
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.conv1d = Conv1d(num_channels, num_filters * 2, filter_size, stride, padding, dilation)

    def forward(self, seq, mask=None):
        h = self.conv1d(seq)
        g, h = h[:, :, :self.num_filters], h[:, :, self.num_filters:]
        if self.dropout_rate:
            g = layers.dropout(g, dropout_prob=self.dropout_rate, dropout_implementation="upscale_in_train",
                               is_test=not self.training)
        g = layers.sigmoid(g)
        seq = g * seq + (1 - g) * h
        if mask is not None:
            seq = seq * mask
        return seq


class MultiHeadAttention(fluid.dygraph.Layer):

    def __init__(self, input_dim, d_key=None, d_value=None, n_head=1, dropout_rate=0.):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_key = input_dim // n_head if d_key is None else d_value
        self.d_value = input_dim // n_head if d_value is None else d_value
        self.dropout_rate = dropout_rate
        self.q_proj = nn.Linear(input_dim, self.d_key * self.n_head)
        self.k_proj = nn.Linear(input_dim, self.d_key * self.n_head)
        self.v_proj = nn.Linear(input_dim, self.d_value * self.n_head)

    def forward(self, queries, keys=None, values=None, mask=None):
        keys = queries if keys is None else keys
        values = keys if values is None else values
        q = self.q_proj(queries)
        k = self.q_proj(keys)
        v = self.q_proj(values)
        q = layers.transpose(layers.reshape(q, shape=[0, 0, self.n_head, self.d_key]), [0, 2, 1, 3])
        k = layers.transpose(layers.reshape(k, shape=[0, 0, self.n_head, self.d_key]), [0, 2, 1, 3])
        v = layers.transpose(layers.reshape(v, shape=[0, 0, self.n_head, self.d_value]), [0, 2, 1, 3])
        scaled_q = layers.scale(x=q, scale=self.d_key ** -0.5)
        product = layers.matmul(x=scaled_q, y=k, transpose_y=True)
        if mask is not None:
            product -= (1 - layers.transpose(layers.unsqueeze(mask, 1), [0, 1, 3, 2])) * 1e10
        weights = layers.softmax(product)
        if self.dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=self.dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=not self.training)
        out = layers.matmul(weights, v)
        out = layers.reshape(layers.transpose(out, [0, 2, 1, 3]), [0, 0, self.d_value * self.n_head])

        return out


class ERModel(fluid.dygraph.Layer):

    def __init__(self, t_dim):
        super(ERModel, self).__init__()
        self.dilated_gated_conv1d_list = []
        for i in range(3):
            self.dilated_gated_conv1d_list.append(
                DilatedGatedConv1d(t_dim, t_dim, filter_size=3, padding=1, dilation=1))
            self.dilated_gated_conv1d_list.append(
                DilatedGatedConv1d(t_dim, t_dim, filter_size=3, padding=2, dilation=2))
            self.dilated_gated_conv1d_list.append(
                DilatedGatedConv1d(t_dim, t_dim, filter_size=3, padding=5, dilation=5))
        self.dilated_gated_conv1d_list.append(DilatedGatedConv1d(t_dim, t_dim, filter_size=3, padding=1, dilation=1))
        self.dilated_gated_conv1d_list.append(DilatedGatedConv1d(t_dim, t_dim, filter_size=3, padding=1, dilation=1))
        self.dilated_gated_conv1d_list.append(DilatedGatedConv1d(t_dim, t_dim, filter_size=3, padding=1, dilation=1))

        self.pn1_fc1 = nn.Linear(t_dim, t_dim)
        self.pn1_fc2 = nn.Linear(t_dim, 1)

        self.pn2_fc1 = nn.Linear(t_dim, t_dim)
        self.pn2_fc2 = nn.Linear(t_dim, 1)

        self.multi_head_attn = MultiHeadAttention(t_dim, n_head=8)

        self.conv1d = Conv1d(t_dim * 2, t_dim, filter_size=3, padding=1)

        self.ps1_fc = nn.Linear(t_dim, 1)
        self.ps2_fc = nn.Linear(t_dim, 1)

    def forward(self, t, mask=None):
        for dilated_gated_conv1d in self.dilated_gated_conv1d_list:
            t = dilated_gated_conv1d(t)

        pn1 = layers.relu(self.pn1_fc1(t))
        pn1 = layers.sigmoid(self.pn1_fc2(pn1))

        pn2 = layers.relu(self.pn2_fc1(t))
        pn2 = layers.sigmoid(self.pn2_fc2(pn2))

        h = self.multi_head_attn(t, mask=mask)

        h = layers.concat([t, h], axis=-1)

        h = self.conv1d(h)

        h = layers.relu(h)

        ps1 = layers.sigmoid(self.ps1_fc(h)) * pn1

        ps2 = layers.sigmoid(self.ps2_fc(h)) * pn2

        return t, pn1, pn2, ps1, ps2


class REModel(fluid.dygraph.Layer):

    def __init__(self, t_dim, num_class):
        super(REModel, self).__init__()
        self.pc_fc1 = nn.Linear(t_dim, t_dim)
        self.pc_fc2 = nn.Linear(t_dim, num_class)

        self.multi_head_attn = MultiHeadAttention(t_dim, n_head=8)

        self.conv1d = Conv1d(t_dim * 4, t_dim, filter_size=3, padding=1)

        self.po_fc = nn.Linear(t_dim, 1)
        self.po1_fc = nn.Linear(t_dim, num_class)
        self.po2_fc = nn.Linear(t_dim, num_class)

    def forward(self, t, k, pn1, pn2, mask=None):
        pc = self._compute_pc(t, mask)

        k = get_k_inter(t, k)
        k = layers.expand(k, [1, t.shape[1], 1])

        h = self.multi_head_attn(t, mask=mask)
        h = layers.concat([t, h, k], axis=-1)
        h = self.conv1d(h)

        po = layers.sigmoid(self.po_fc(h))
        po1 = layers.sigmoid(self.po1_fc(h))
        po2 = layers.sigmoid(self.po2_fc(h))

        po1 = po * po1 * pc * pn1
        po2 = po * po2 * pc * pn2

        return po1, po2

    def _compute_pc(self, x, mask):
        if mask is not None:
            x -= (1 - mask) * 1e10
        x = layers.reduce_max(x, dim=1, keep_dim=True)
        x = layers.relu(self.pc_fc1(x))
        x = layers.sigmoid(self.pc_fc2(x))
        return x


class KGModel(fluid.dygraph.Layer):

    def __init__(self, t_dim, maxlen, char_num, word_num, word_dim, word_vec, num_class,
                 padding_idx=0, dropout_rate=0.25):
        super(KGModel, self).__init__()
        self.pe = nn.Embedding(size=[maxlen, t_dim],
                               param_attr=fluid.ParamAttr(name="position_embedding.w_0",
                                                          initializer=fluid.initializer.ConstantInitializer(
                                                              value=0.)))

        self.ce = nn.Embedding(size=[char_num, t_dim], padding_idx=padding_idx,
                               param_attr=fluid.ParamAttr(name="char_embedding.w_0"))

        self.we_p = nn.Embedding(size=[word_num, word_dim], padding_idx=padding_idx,
                                 param_attr=fluid.ParamAttr(name="word_embedding.w_0",
                                                            initializer=fluid.initializer.NumpyArrayInitializer(
                                                                word_vec),
                                                            trainable=False))

        self.we = nn.Linear(word_dim, t_dim, param_attr=fluid.ParamAttr(name="word_embedding.w_1"), bias_attr=False)

        self.er_model = ERModel(t_dim)

        self.re_model = REModel(t_dim, num_class)

        self.padding_idx = padding_idx

        self.dropout_rate = dropout_rate

    def embedding(self, t1, t2, mask=None):
        pv = self.pe(position_id(t1))
        t1 = self.ce(t1)
        t2 = self.we(self.we_p(t2))
        t = t1 + t2 + pv
        if self.dropout_rate:
            t = layers.dropout(
                t,
                dropout_prob=self.dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=not self.training)
        if mask is not None:
            t = t * mask
        return t

    def forward(self, t1, t2, k):
        mask = get_mask(t1, padding_idx=self.padding_idx)

        t = self.embedding(t1, t2)

        t, pn1, pn2, ps1, ps2 = self.er_model(t, mask=mask)

        po1, po2 = self.re_model(t, k, pn1, pn2, mask=mask)

        return ps1, ps2, po1, po2, mask


def loss_func(ps1, ps2, s1, s2, po1, po2, o1, o2, mask):
    ps1 = layers.concat([1 - ps1, ps1], axis=-1)
    ps2 = layers.concat([1 - ps2, ps2], axis=-1)
    s1 = layers.unsqueeze(s1, -1)
    s2 = layers.unsqueeze(s2, -1)

    s1_loss = layers.cross_entropy(ps1, s1)
    s1_loss = layers.reduce_sum(s1_loss * mask) / layers.reduce_sum(mask)

    s2_loss = layers.cross_entropy(ps2, s2)
    s2_loss = layers.reduce_sum(s2_loss * mask) / layers.reduce_sum(mask)

    po1, o1 = layers.unsqueeze(po1, -1), layers.unsqueeze(o1, -1)
    po1 = layers.concat([1 - po1, po1], axis=-1)
    o1_loss = layers.reduce_sum(layers.cross_entropy(po1, o1), 2)
    o1_loss = layers.reduce_sum(o1_loss * mask) / layers.reduce_sum(mask)

    po2, o2 = layers.unsqueeze(po2, -1), layers.unsqueeze(o2, -1)
    po2 = layers.concat([1 - po2, po2], axis=-1)
    o2_loss = layers.reduce_sum(layers.cross_entropy(po2, o2), 2)
    o2_loss = layers.reduce_sum(o2_loss * mask) / layers.reduce_sum(mask)

    loss = (s1_loss + s2_loss) + (o1_loss + o2_loss)

    return loss
