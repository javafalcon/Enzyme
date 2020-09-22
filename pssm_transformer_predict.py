from Bio import SeqIO
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import re
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tools import displayMetrics, displayMLMetrics, plot_history
from ncbi import create_padding_pssm_mask

"""
位置函数
    基于角度的位置编码方法。计算位置编码矢量的长度
    Parameters
    ----------
    pos : 
        在句子中字的位置序号，取值范围是[0, max_sequence_len).
    i   : int
        字向量的维度，取值范围是[0, embedding_dim).
    embedding_dim : int
        字向量最大维度， 即embedding_dim的最大值.

    Returns
    -------
    float32
        第pos位置上对应矢量的长度.
"""
def get_angles(pos, i, embed_dim):
    angel_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
    return pos * angel_rates

# 位置编码
def position_encoding(position, embed_dim):
    angel_rads = get_angles(np.arange(position)[:, np.newaxis], 
                            np.arange(embed_dim)[np.newaxis, :], 
                            embed_dim)
    sines = np.sin(angel_rads[:, 0::2])
    cones = np.cos(angel_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1)
    #pos_encoding = pos_encoding[np.newaxis, ...]
    return pos_encoding


# 自注意力机制
def scaled_dot_product_attention(q, k, v, mask):
    
    # query key 相乘获取匹配关系
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # 掩码.将被掩码的token乘以-1e9（表示负无穷），这样
    # softmax之后就为0， 不对其它token产生影响
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # 通过softmax获取attention权重
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # attention乘上value
    output = tf.matmul(attention_weights, v) # (..., seq_len_v, depth)
    
    return output, attention_weights

class MultiHeadAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        # embd_dim必须可以被num_heads整除
        assert embed_dim % num_heads == 0
        # 分头后的维度
        self.projection_dim = embed_dim // num_heads
        self.wq = layers.Dense(embed_dim)
        self.wk = layers.Dense(embed_dim)
        self.wv = layers.Dense(embed_dim)
        self.dense = layers.Dense(embed_dim)
        
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        # 分头前的前向网络，获取q, k, v语义
        q = self.wq(q)
        k = self.wq(k)
        v = self.wv(v)
        
        # 分头
        q = self.separate_heads(q, batch_size) # [batch_size, num_heads, seq_len_q, projection_dim]
        k = self.separate_heads(k, batch_size)
        v = self.separate_heads(v, batch_size)
        
        # 通过缩放点积注意力层
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])
        # 合并多头
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embed_dim))
        
        # 全连接重塑
        output = self.dense(concat_attention)
        
        return output, attention_weights
    
# 构造前向网络
def point_wise_feed_forward_network(d_model, diff):
    # d_model 即embed_dim
    return tf.keras.Sequential([
        layers.Dense(diff, activation='relu'),
        layers.Dense(d_model)])    
    
# transformer编码层
class EncoderLayer(layers.Layer):
    def __init__(self, d_model, n_heads, ffd, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = point_wise_feed_forward_network(d_model, ffd)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training, mask):
        att_output, _ = self.mha(inputs, inputs, inputs, mask)
        att_output = self.dropout1(att_output, training=training)
        out1 = self.layernorm1(inputs + att_output)
        # 前向网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class Encoder(layers.Layer):
    def __init__(self, n_layers, d_model, n_heads, ffd,
                 input_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Encoder, self).__init__()
        
        self.n_layers = n_layers
        self.d_model = d_model
        self.pos_embedding = position_encoding(max_seq_len, d_model)
        self.encoder_layer = [EncoderLayer(d_model, n_heads, ffd, dropout_rate)
                              for _ in range(n_layers)]
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training, mask):
        word_emb = tf.cast(inputs, tf.float32)
        #word_emb *= (tf.cast(self.d_model, tf.float32))
        emb = word_emb + self.pos_embedding
        x = self.dropout(emb, training=training)
        for i in range(self.n_layers):
            x = self.encoder_layer[i](x, training, mask)
        return x
    
def buildModel(maxlen, vocab_size, embed_dim, num_heads, ff_dim, 
               num_blocks, droprate, fl_size, num_classes):
    inputs = layers.Input(shape=(maxlen,20))
    masks = layers.Input(shape=(1,1,maxlen))
    
    encoder = Encoder(n_layers=num_blocks, d_model=embed_dim, n_heads=num_heads, 
                      ffd=ff_dim, input_vocab_size=vocab_size, 
                      max_seq_len=maxlen, dropout_rate=droprate)
    x = encoder(inputs, True, masks)
    
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(droprate)(x)
    x = layers.Dense(fl_size, activation="relu")(x)
    x = layers.Dropout(droprate)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=[inputs, masks], outputs=outputs)
    
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    return model    

def transformer_predictor(X_train, y_train, m_train, m_test, X_test, y_test, modelfile, params):
    keras.backend.clear_session()

    model = buildModel(params['maxlen'], params['vocab_size'], params['embed_dim'], 
                    params['num_heads'], params['ff_dim'],  params['num_blocks'], 
                    params['droprate'], params['fl_size'], params['num_classes'])
    model.summary()

    checkpoint = callbacks.ModelCheckpoint(modelfile, monitor='val_loss',
                                       save_best_only=True, 
                                       save_weights_only=True, 
                                       verbose=1)
    history = model.fit(
        [X_train, m_train], y_train, 
        batch_size=params['batch_size'], epochs=params['epochs'], 
        validation_data=([X_test, m_test], y_test),
        callbacks=[checkpoint]
        )

    plot_history(history)

    #model.load_weights(modelfile)
    score = model.predict([X_test, m_test])
    
    return score

# transformer net params
params = {}
params['vocab_size'] = 24
params['maxlen'] = 500
params['embed_dim'] = 20 # Embedding size for each token
params['num_heads'] = 4  # Number of attention heads
params['ff_dim'] = 128  # Hidden layer size in feed forward network inside transformer
params['num_blocks'] = 12
params['droprate'] = 0.2
params['fl_size'] = 128
params['num_classes'] = 2
params['epochs'] = 20
params['batch_size'] = 32


def load_pssm(dirname, num_prots=None):
    listf = os.listdir(dirname)
    if num_prots is not None:
        listf = shuffle(listf)
        listf = listf[:num_prots]
    num_len_ec = len(listf)
    
    pssm = np.ndarray(shape=(num_len_ec, params["maxlen"], 20))
    mask = np.ndarray(shape=(num_len_ec, params["maxlen"]))
    for i in range(num_len_ec):
        x,m = create_padding_pssm_mask(os.path.join(dirname, listf[i]), maxlen=params["maxlen"])
        pssm[i, :, :] = x
        mask[i,:] = m
        
    return pssm, mask

# load data
"""
ec_pssm, ec_mask = load_pssm("e:/Repoes/Enzyme/pssm/ec")
np.savez('ec_pssm_{}.npz'.format(params['maxlen']), pssm=ec_pssm, mask=ec_mask)

notec_pssm, notec_mask = load_pssm("e:/Repoes/Enzyme/pssm/not_ec")
np.savez('notec_pssm_{}.npz'.format(params['maxlen']), pssm=notec_pssm, mask=notec_mask)
"""
ec_data = np.load('ec_pssm_500.npz')
ec_pssm, ec_mask = ec_data['pssm'], ec_data['mask']
notec_data = np.load('notec_pssm_500.npz')
notec_pssm, notec_mask = notec_data['pssm'], notec_data['mask']

pssm = np.concatenate((ec_pssm, notec_pssm))
mask = np.concatenate((ec_mask, notec_mask))
mask = mask[:,np.newaxis, np.newaxis,:]
labels = [1 for _ in range(ec_pssm.shape[0])] + [0 for _ in range(notec_pssm.shape[0])]

# split data into train and test
x_train, x_test, m_train, m_test, labels_train, labels_test = train_test_split(pssm, mask, labels, 
                                                    test_size=0.3, 
                                                    random_state=42,
                                                    stratify=labels)



y_train = keras.utils.to_categorical(labels_train, params['num_classes'])
y_test = keras.utils.to_categorical(labels_test, params['num_classes'])

# training and test
modelfile = './model/ec/ec_trainsformer_{}_{}.h5'.format(params["maxlen"], "pos")
score = transformer_predictor(x_train, y_train, m_train, m_test, x_test, y_test, modelfile, params)
pred = np.argmax(score, 1)
displayMetrics(np.argmax(y_test, 1), pred)


