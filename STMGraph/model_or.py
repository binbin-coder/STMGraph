import tensorflow.compat.v1 as tf

class SDGATE():

    def __init__(self, hidden_dims, alpha=0.8, nonlinear=True, weight_decay=0.0001):
        self.n_layers = len(hidden_dims) - 1
        self.alpha = alpha
        self.W, self.v,self.learnable_param = self.define_weights(hidden_dims)
        self.C = {}
        self.nonlinear = nonlinear
        self.weight_decay = weight_decay

    def sce_loss(self,x, y, alpha=1):
        x = tf.math.l2_normalize(x, axis=-1)
        y = tf.math.l2_normalize(y, axis=-1)

        loss = tf.pow(1 - tf.reduce_sum(tf.multiply(x, y), axis=-1), alpha)
        # loss = tf.reduce_mean(loss)
        loss = tf.reduce_sum(loss)
        return loss

    # 生成行屏蔽的掩码
    def create_mask_matrix(self, X, drop_ratio=0.5,noise_ratio=0.1):
        total_rows = tf.shape(X)[0]
        total_rows = tf.cast(total_rows, tf.float32)

        num_drops = tf.cast(total_rows * drop_ratio, tf.float32)  # 需要置零的行数
        total_rows = tf.cast(total_rows, tf.int32)
        indices = tf.random.shuffle(tf.range(total_rows))  # 随机打乱行索引
        # 创建一个可学习的参数，形状与 drop_indices 的长度相同
        # learnable_param = tf.Variable(tf.zeros((1,)), trainable=True, name="learnable_param")
        
        noise_num_drops = tf.cast(num_drops * noise_ratio, tf.float32)  # 需要置替换噪声的行数
        token_num_drops = num_drops - noise_num_drops
        noise_num_drops = tf.cast(noise_num_drops, tf.int32)
        token_num_drops = tf.cast(token_num_drops, tf.int32)

        num_drops = tf.cast(num_drops, tf.int32)
        drop_indices = tf.gather(indices, tf.range(num_drops), axis=0)  # 选取需要置零的行的索引
        shuffled_drop_indices = tf.random.shuffle(drop_indices)  # 随机打乱行索引
        shuffled_indices = tf.random.shuffle(indices)  # 随机打乱行索引
        token_indices = shuffled_drop_indices[:token_num_drops]
        noise_indices = shuffled_drop_indices[-noise_num_drops:]
        # 创建掩码
        mask = tf.ones(total_rows, dtype=tf.float32)
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(token_indices, axis=1), tf.zeros(token_num_drops))
        # 对 X 矩阵进行屏蔽
        masked_X = X * tf.expand_dims(mask, axis=1)  # 使用 broadcasting 将掩码应用到整个矩阵的每一行
        # 获取需要替换噪声的行的索引
        noise_indices_M = shuffled_indices[-noise_num_drops:]
        # 从 X 矩阵中随机选择与 noise_indices 相同行数的行，并替换到 masked_X 中对应的行
        selected_noise = tf.gather(X, noise_indices_M)
        masked_X = tf.tensor_scatter_nd_update(masked_X, tf.expand_dims(noise_indices, axis=1), selected_noise)
        masked_X = X * tf.expand_dims(mask, axis=1)  # 使用 broadcasting 将掩码应用到整个矩阵的每一行
        masked_rows = tf.gather(masked_X, token_indices)
        # 将 learnable_param 加到 masked_rows 中
        updated_rows = masked_rows + tf.expand_dims(self.learnable_param, axis=1)
        masked_X = tf.tensor_scatter_nd_update(masked_X, tf.expand_dims(token_indices, axis=1), updated_rows)
        return masked_X, drop_indices, num_drops

    def re_mask(self, X, drop_indices, num_drops):
        total_rows = tf.shape(X)[0]
        total_rows = tf.cast(total_rows, tf.int32)
        mask = tf.ones(total_rows, dtype=tf.float32)
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(drop_indices, axis=1), tf.zeros(num_drops))
        masked_X = X * tf.expand_dims(mask, axis=1)  # 使用 broadcasting 将掩码应用到整个矩阵的每一行
        return masked_X

    def __call__(self, A, X, mask_ratio, noise):

        # H,drop_indices, num_drops=self.create_mask_matrix(X,drop_ratio=dropout,noise_ratio=noise)
        # Encoder
        H = X
        for layer in range(self.n_layers):
            H = self.__encoder(A, H, layer)
            if self.nonlinear:
                if layer != self.n_layers - 1:
                    H = tf.nn.elu(H)
        # Final node representations
        self.H = H
        # H = self.re_mask(H, drop_indices, num_drops)
        # Decoder
        for layer in range(self.n_layers - 1, -1, -1):
            H = self.__decoder(H, layer)
            if self.nonlinear:
                if layer != 0:
                    H = tf.nn.elu(H)
        X_ = H
        #H1 = H
        #for layer in range(self.n_layers):
        #    H1 = self.__encoder(A, H1, layer)
        #    if self.nonlinear:
        #        if layer != self.n_layers - 1:
        #            H1 = tf.nn.elu(H1)
        # Final node representations
        #self.H1 = H1
        # The reconstruction loss of node features 
        features_loss = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(X - X_, 2))))

        #Xdrop = tf.gather(X, drop_indices)
        #X_drop = tf.gather(X_, drop_indices)
        # features_loss = self.sce_loss(X, X_)
        # for layer in range(self.n_layers):
        weight_decay_loss = 0
        # for layer in range(self.n_layers):    
        #     weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W[layer][0]), self.weight_decay, name='weight_loss_0')        
        #     weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W[layer][1]), self.weight_decay, name='weight_loss_1') 
        # Total loss
        weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W[self.n_layers-1][0]), self.weight_decay, name='weight_loss_0')
        weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W[self.n_layers-1][1]), self.weight_decay, name='weight_loss_1')
        self.loss = features_loss + weight_decay_loss
        self.Att_l = self.C
        return self.loss, self.H, self.Att_l, X_

    def __encoder(self, A, H, layer):
        H1 = tf.matmul(H, self.W[layer][0])
        H2 = tf.matmul(H, self.W[layer][1])
        H = tf.add(H1, H2)
        if layer == self.n_layers - 1:
            return H
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)


    def __decoder(self, H, layer):
        # if layer>0:
        # H1 = tf.add(tf.matmul(H, self.W[layer][0], transpose_b=True),self.b_d0[layer-1][0])
        # H2 = tf.add(tf.matmul(H, self.W[layer][1], transpose_b=True),self.b_d0[layer-1][1])
        # else:
        H1 = tf.matmul(H, self.W[layer][0], transpose_b=True)
        H2 = tf.matmul(H, self.W[layer][1], transpose_b=True)
        H = tf.add(H1, H2)
        if layer == 0:
            return H
        return tf.sparse_tensor_dense_matmul(self.C[layer - 1], H)

    def define_weights(self, hidden_dims):
        W_d = {}
        for i in range(self.n_layers):
            W = {}
            W[0] = tf.get_variable("W%s_0" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))
            W[1] = tf.get_variable("W%s_1" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))
            W_d[i] = W

        Ws_att = {}
        for i in range(self.n_layers - 1):
            Ws_att[i] = tf.get_variable("v%s" % i, shape=(hidden_dims[i + 1], 1))
        learnable_param = tf.Variable(tf.zeros((1,)), trainable=True,name = "learnable_param")
        return W_d, Ws_att, learnable_param

    def graph_attention_layer(self, A, M, v, layer):
        with tf.variable_scope("layer_%s" % layer):
            f1 = tf.matmul(M, v)

            f1 = A * f1
            unnormalized_attentions1 = tf.SparseTensor(indices=f1.indices,
                                                       values=tf.nn.sigmoid(f1.values),
                                                       dense_shape=f1.dense_shape)
            unnormalized_attentions2 = tf.sparse_transpose(unnormalized_attentions1, [1, 0])
            unnormalized_attentions = tf.sparse_add(unnormalized_attentions1, unnormalized_attentions2)
            attentions = tf.sparse_softmax(unnormalized_attentions)
            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)
            return attentions

