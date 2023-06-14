import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense, Conv1D, Dropout, LayerNormalization
import math



class MultiHeadAttention(Layer):
    '''
    Multi-Head Attention
    '''
    def __init__(self, hidden_dim, heads_num, drop_rate=0.5):
        super(MultiHeadAttention, self).__init__()
        # 入力の線形変換
        # 重み行列は[hidden_dim, hidden_dim]
        self.query = Conv1D(hidden_dim, kernel_size=1)
        self.key   = Conv1D(hidden_dim, kernel_size=1)
        self.value = Conv1D(hidden_dim, kernel_size=1)
        
        # 出力の線形変換
        self.projection = Conv1D(hidden_dim, kernel_size=1)
        
        # 出力のDropout
        self.drop = Dropout(drop_rate)
        
        self.nf = hidden_dim
        self.nh = heads_num
    
    def atten(self, query, key, value, attention_mask, training):
        """
        Attention
        
        query, key, value : 入力
        attention_mask : attention weight に適用される mask
        """
        # 各値を取得
        shape = query.shape.as_list()
        batch_size = -1 if shape[0] is None else shape[0]
        token_num = shape[2] # トークン列数
        hidden_dim = shape[1]*shape[3] # 入力チャンネル数

        scores = tf.matmul(query, key, transpose_b=True)
        scores = tf.multiply(scores, tf.math.rsqrt(tf.cast(hidden_dim, tf.float32)))
        if attention_mask is not None:
            scores += attention_mask * -1e9

        atten_weight = tf.nn.softmax(scores)
        context = tf.matmul(atten_weight, value)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, token_num, hidden_dim))
        context = self.projection(context, training=training)
        
        return self.drop(context, training=training), atten_weight

    def _split(self, x):
        """
        query, key, valueを分割する
        
        入力 shape: [batch_size, length, hidden_dim] の時
        出力 shape: [batch_size, head_num, length, hidden_dim//head_num]
        """
        # 各値を取得
        hidden_dim = self.nf
        heads_num = self.nh
        shape = x.shape.as_list()
        batch_size = -1 if shape[0] is None else shape[0]
        token_num = shape[1] # トークン列数
        
        # [batch_size, token_num, hidden_dim] -> [batch_size, token_num, head_num, hidden_dim/head_num]
        # splitだが実際は次元を拡張する処理
        x = tf.reshape(x, (batch_size, token_num, heads_num, int(hidden_dim/heads_num)))
        
        # [batch_size, token_num, head_num, hidden_dim/head_num] -> [batch_size, head_num, token_num, hidden_dim/head_num]
        x = tf.transpose(x, [0, 2, 1, 3])
        return x
    
    def call(self, x, training, memory=None, attention_mask=None, return_attention_scores=False):
        """
        モデルの実行
        """
        if memory is None:
            memory = x
        
        # input -> query
        # memory -> key, value
        # [batch_size, token_num, hidden_dim] @ [hidden_dim, hidden_dim] -> [batch_size, token_num, hidden_dim] 
        query = self.query(x)
        key = self.key(memory)
        value = self.value(memory)
        
        # [batch_size, token_num, hidden_dim] -> [batch_size, head_num, token_num, hidden_dim/head_num]
        query = self._split(query)
        key = self._split(key)
        value = self._split(value)
        
        # attention
        # context: [batch_size, token_num, hidden_dim]
        context, attn_weights = self.atten(query, key, value, attention_mask, training)
        if not return_attention_scores:
            return context
        else:
            return context, attn_weights
        
        
class FeedForwardNetwork(Layer):
    '''
    Position-wise Feedforward Neural Network
    transformer blockで使用される全結合層
    '''
    def __init__(self, hidden_dim, drop_rate):
        super().__init__()
        # 1層目：チャンネル数を増加させる
        self.filter_dense_layer = Dense(hidden_dim * 4, use_bias=True, activation='relu')
        
        # 2層目：元のチャンネル数に戻す
        self.output_dense_layer = Dense(hidden_dim, use_bias=True)
        self.drop = Dropout(drop_rate)

    def call(self, x, training):
        '''
        [batch_size, token_num, hidden_dim]
        '''
        
        # [batch_size, token_num, hidden_dim] -> [batch_size, token_num, 4*hidden_dim]
        x = self.filter_dense_layer(x)
        x = self.drop(x, training=training)
        
        # [batch_size, token_num, 4*hidden_dim] -> [batch_size, token_num, hidden_dim]
        return self.output_dense_layer(x)
    
    
class ResidualNormalizationWrapper(Layer):
    '''
    残差接続
    output: input + SubLayer(input)
    '''
    def __init__(self, layer, drop_rate):
        super().__init__()
        self.layer = layer # SubLayer : ここではAttentionかFFN
        self.layer_normalization = LayerNormalization()
        self.drop = Dropout(drop_rate)

    def call(self, x, training, memory=None, attention_mask=None, return_attention_scores=None):
        """
        AttentionもFFNも入力と出力で形が変わらない
        [batch_size, token_num, hidden_dim]
        """
        
        params = {}
        if memory is not None:
            params['memory'] = memory
        if attention_mask is not None:
            params['attention_mask'] = attention_mask
        if return_attention_scores:
            params['return_attention_scores'] = return_attention_scores
        
        out = self.layer_normalization(x)
        if return_attention_scores:
            out, attn_weights = self.layer(out, training, **params)
            out = self.drop(out, training=training)
            return x + out, attn_weights
        else:
            out = self.layer(out, training, **params)
            out = self.drop(out, training=training)
            return x + out
        
        
class AddPositionalEncoding(Layer):
    '''
    入力テンソルに対し、位置の情報を付与して返すレイヤー
    see: https://arxiv.org/pdf/1706.03762.pdf

    PE_{pos, 2i}   = sin(pos / 10000^{2i / d_model})
    PE_{pos, 2i+1} = cos(pos / 10000^{2i / d_model})
    '''
    def call(self, inputs):
        fl_type = inputs.dtype
        batch_size, max_length, depth = tf.unstack(tf.shape(inputs))

        depth_counter = tf.range(depth) // 2 * 2  # 0, 0, 2, 2, 4, ...
        depth_matrix = tf.tile(tf.expand_dims(depth_counter, 0), [max_length, 1])  # [max_length, depth]
        depth_matrix = tf.pow(10000.0, tf.cast(depth_matrix / depth, fl_type))  # [max_length, depth]

        # cos(x) == sin(x + π/2)
        phase = tf.cast(tf.range(depth) % 2, fl_type) * math.pi / 2  # 0, π/2, 0, π/2, ...
        phase_matrix = tf.tile(tf.expand_dims(phase, 0), [max_length, 1])  # [max_length, depth]

        pos_counter = tf.range(max_length)
        pos_matrix = tf.cast(tf.tile(tf.expand_dims(pos_counter, 1), [1, depth]), fl_type)  # [max_length, depth]

        positional_encoding = tf.sin(pos_matrix / depth_matrix + phase_matrix)
        # [batch_size, max_length, depth]
        positional_encoding = tf.tile(tf.expand_dims(positional_encoding, 0), [batch_size, 1, 1])

        return inputs + positional_encoding
    
    
class TransformerBlock(Layer):
    """
    transformer block : before ->[attention -> FF]-> next
    それぞれ残差接続とLayerNormalizationの処理が含まれる
    """
    def __init__(self, hidden_dim, heads_num, drop_rate=0.1):
        """
        hidden_numはheads_numで割り切れえる値とすること
        """
        super().__init__()
        self.atten = ResidualNormalizationWrapper(
            layer = MultiHeadAttention(hidden_dim = hidden_dim, heads_num = heads_num, drop_rate = drop_rate),
            drop_rate = drop_rate)
        
        self.ffn = ResidualNormalizationWrapper(
            layer = FeedForwardNetwork(hidden_dim = hidden_dim, drop_rate = drop_rate),
            drop_rate = drop_rate)
    
    def call(self, input, training, memory=None, attention_mask=None, return_attention_scores=False):
        """
        入力と出力で形式が変わらない
        [batch_size, token_num, hidden_dim]
        """
        
        if return_attention_scores:
            x, attn_weights = self.atten(input,training, memory, attention_mask, return_attention_scores)
            x = self.ffn(x)
            return x, attn_weights
        else:
            x = self.atten(input, training, memory, attention_mask, return_attention_scores)
            x = self.ffn(x)
            return x
        
        
        
class TransformerEncoder(Layer):
    '''
    TransformerのEncoder
    '''
    def __init__(
            self,
            input_dim,
            hopping_num, # Multi-head Attentionの繰り返し数
            heads_num, # Multi-head Attentionのヘッド数
            hidden_dim, # Embeddingの次数
            drop_rate, # ドロップアウトの確率
            embeddings=None
    ):
        super().__init__()
        self.hopping_num = hopping_num
        
        # Position Embedding
        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = Dropout(drop_rate)

        # Multi-head Attentionの繰り返し(hopping)のリスト
        self.attention_block_list = [TransformerBlock(hidden_dim, heads_num) for _ in range(hopping_num)]
        self.output_normalization = LayerNormalization()

    def call(
            self,
            inputs,
            training,
            memory=None,
            attention_mask=None,
            return_attention_scores=False
    ):
        '''
        input: 入力 [batch_size, length]
        memory: 入力 [batch_size, length]
        attention_mask: attention weight に適用される mask
            [batch_size, 1, q_length, k_length] 
            pad 等無視する部分が 0 となるようなもの(Decoderで使用)
        return_attention_scores : attention weightを出力するか
        出力 [batch_size, length, hidden_dim]
        '''
        # [batch_size, token_num, hidden_dim]
        embedded_input = inputs
        # Positional Embedding
        embedded_input = self.add_position_embedding(embedded_input)
        query = self.input_dropout_layer(embedded_input, training=training)
        
        
        if return_attention_scores:
            # MultiHead Attentionを繰り返し適用
            for i in range(self.hopping_num):
                query, atten_weights = self.attention_block_list[i](query, training, memory, attention_mask, return_attention_scores)

            # [batch_size, token_num, hidden_dim]
            return self.output_normalization(query), atten_weights
        else:
            # MultiHead Attentionを繰り返し適用
            for i in range(self.hopping_num):
                query = self.attention_block_list[i](query, training, memory, attention_mask, return_attention_scores)

            # [batch_size, token_num, hidden_dim]
            return self.output_normalization(query)
        

class TransformerDecoder(Layer):
    def __init__(
            self,
            input_dim, 
            token_num, 
            hopping_num, 
            heads_num, 
            hidden_dim, 
            drop_rate,
            embeddings=None
    ):
        super().__init__()
        self.hopping_num = hopping_num
        self.hidden_dim = hidden_dim
        self.heads_num = heads_num
        
        # Position Embedding
        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = Dropout(drop_rate)
        
        # Multi-head Attentionの繰り返し(hopping)のリスト
        self.attention_block_list1 = [TransformerBlock(hidden_dim, heads_num) for _ in range(hopping_num)]
        self.attention_block_list2 = [TransformerBlock(hidden_dim, heads_num) for _ in range(hopping_num)]
        self.output_normalization = LayerNormalization()

    def call(
            self,
            inputs,
            training,
            memory=None,
            attention_mask=None,
            return_attention_scores=False
    ):
        embedded_input = inputs
        embedded_input = self.add_position_embedding(embedded_input)
        query = self.input_dropout_layer(embedded_input, training=training)
        
        if return_attention_scores:
            for i in range(self.hopping_num):
                query, atten_weights = self.attention_block_list1[i](query, training, memory, attention_mask, return_attention_scores)
                query, atten_weights = self.attention_block_list2[i](query, training, query, attention_mask, return_attention_scores)

            return self.output_normalization(query), atten_weights
        else:
            for i in range(self.hopping_num):
                query = self.attention_block_list1[i](query, training, memory, attention_mask, return_attention_scores)
                query = self.attention_block_list2[i](query, training, query, attention_mask, return_attention_scores)

            return self.output_normalization(query)