import tensorflow.compat.v1 as tf

class SMGATE():

    def __init__(self,hidden_dims,nonlinear=True, weight_decay=0.0001,alpha=1.0):
        self.n_layers = len(hidden_dims) - 1
        self.alpha = alpha
        self.W, self.v, self.learnable_param1, self.learnable_param2 = self.define_weights(hidden_dims)
        self.C = {}
        self.nonlinear = nonlinear
        self.weight_decay = weight_decay

    def sce_loss(self,x, y, alpha=1):
        x = tf.math.l2_normalize(x, axis=-1)
        y = tf.math.l2_normalize(y, axis=-1)

        loss = tf.pow(1 - tf.reduce_sum(tf.multiply(x, y), axis=-1), alpha)
        loss = tf.reduce_mean(loss)
        # loss = tf.reduce_sum(loss)
        return loss

    # Mask to generate row mask
    def create_mask_matrix(self, X, drop_ratio=0.5, noise_ratio=0.05):
        noise_ratio = noise_ratio
        total_rows = tf.shape(X)[0]
        total_rows = tf.cast(total_rows, tf.float32)

        num_drops = tf.cast(total_rows * drop_ratio, tf.float32)  # Number of rows that need to be set to 0
        total_rows = tf.cast(total_rows, tf.int32)
        indices = tf.random.shuffle(tf.range(total_rows))  # Randomly scramble row index

        noise_num_drops = tf.cast(num_drops * noise_ratio, tf.float32)  # Number of noise lines to replace
        noise_num_drops = tf.cast(noise_num_drops, tf.int32)
        num_drops = tf.cast(num_drops, tf.int32)  # Number of rows that need to be set to 0
        token_num_drops = num_drops - noise_num_drops
        token_num_drops = tf.cast(token_num_drops, tf.int32)

        drop_indices = tf.gather(indices, tf.range(num_drops), axis=0)  # Select the index of the row that needs to be zeroed
        shuffled_drop_indices = tf.random.shuffle(drop_indices)
        shuffled_indices = tf.random.shuffle(indices) 
        token_indices = shuffled_drop_indices[:token_num_drops]

        noise_indices = shuffled_drop_indices[token_num_drops:]

        keep_indices = tf.sets.difference(tf.expand_dims(indices, axis=0), tf.expand_dims(drop_indices, axis=0)).values
        keep_indices = tf.cast(keep_indices, tf.int32)

        num_keeps = total_rows - num_drops
        # create masked martix
        mask = tf.ones(total_rows, dtype=tf.float32)
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(token_indices, axis=1), tf.zeros(token_num_drops))
        # Mask the X matrix
        masked_X = X * tf.expand_dims(mask, axis=1)  # Use broadcasting to apply the mask to each row of the entire matrix
        # Get replaced noise row
        noise_indices_M = shuffled_indices[:noise_num_drops]
        # Random selection and noise from X matrix_ Indents lines with the 
        # same number of lines and replaces them with masked_ Corresponding line in X
        selected_noise_M = tf.gather(X, noise_indices_M)
        # print(noise_num_drops, noise_indices_M, noise_indices)
        masked_X = tf.tensor_scatter_nd_update(masked_X, tf.expand_dims(noise_indices, axis=1), selected_noise_M)

        # Get the submatrix of the row that needs to be set to zero
        masked_rows = tf.gather(masked_X, token_indices)
        # learnable_param add to masked_rows
        updated_rows = masked_rows + self.learnable_param1
        # update tf.tensor_scatter_nd_add matrix to masked_X
        masked_X = tf.tensor_scatter_nd_update(masked_X, tf.expand_dims(token_indices, axis=1), updated_rows)
        return masked_X, drop_indices, num_drops ,keep_indices,num_keeps

    def re_mask(self, X, drop_indices, num_drops):
        total_rows = tf.shape(X)[0]
        total_rows = tf.cast(total_rows, tf.int32)
        mask = tf.ones(total_rows, dtype=tf.float32)
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(drop_indices, axis=1), tf.zeros(num_drops))
        masked_X = X * tf.expand_dims(mask, axis=1)  # Broadcasting applies a mask to each row of the entire matrix
        return masked_X

    def re_random_mask(self, X, num_drops):
        total_rows = tf.shape(X)[0]
        total_rows = tf.cast(total_rows, tf.int32)
        mask = tf.ones(total_rows, dtype=tf.float32)
        shuffle_indices = tf.random.shuffle(tf.range(total_rows))
        drop_indices = tf.gather(shuffle_indices, tf.range(num_drops), axis=0)
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(drop_indices, axis=1), tf.zeros(num_drops))
        masked_X = X * tf.expand_dims(mask, axis=1)  # Broadcasting applies a mask to each row of the entire matrix
        masked_rows = tf.gather(masked_X, drop_indices)
        updated_rows = masked_rows + self.learnable_param2
        masked_X = tf.tensor_scatter_nd_update(masked_X, tf.expand_dims(drop_indices, axis=1), updated_rows)
        return masked_X

    def __call__(self, A, X, mask_ratio, noise):
        #H0 = X
        #for layer in range(self.n_layers):
        #    H0 = self.__encoder(A, H0, layer)
        #    if self.nonlinear:
        #        if layer != self.n_layers - 1:
        #            H0 = tf.nn.elu(H0)
 
        H_r,drop_indices, num_drops,keep_indices,num_keeps=self.create_mask_matrix(X,drop_ratio=mask_ratio,noise_ratio=noise)
        # Encoder
        # H = X
        for layer in range(self.n_layers):
            H_r = self.__encoder(A, H_r, layer)
            if self.nonlinear:
                if layer != self.n_layers - 1:
                    H_r = tf.nn.elu(H_r)
        #latent_loss = self.sce_loss(H0,H_r)
        # Final node representations
        self.H = H_r
        #H_ = H_r
        # Decoder
        #for layer in range(self.n_layers - 1, -1, -1):
        #    H_ = self.__decoder(H_, layer)
        #    if self.nonlinear:
        #        if layer != 0:
        #            H_ = tf.nn.elu(H_)
        #self.H_ = H_
        #input_loss = self.sce_loss(X,self.H_)
        H_m0 = self.re_random_mask(H_r, num_drops)
        H_m1 = self.re_random_mask(H_r, num_keeps)
        # Decoder
        for layer in range(self.n_layers - 1, -1, -1):
            H_m0 = self.__decoder(H_m0, layer)
            if self.nonlinear:
                if layer != 0:
                    H_m0 = tf.nn.elu(H_m0)
        self.H_m0 = H_m0
        for layer in range(self.n_layers - 1, -1, -1):
            H_m1 = self.__decoder(H_m1, layer)
            if self.nonlinear:
                if layer != 0:
                    H_m1 = tf.nn.elu(H_m1)
        # self.H_m1 = H_m1
        Xdrop = tf.gather(X, drop_indices)
        Xkeep = tf.gather(X, keep_indices)
        H_m0_drop = tf.gather(H_m0, drop_indices)
        H_m1_keep = tf.gather(H_m1, keep_indices)
        features_loss = self.sce_loss(Xdrop, H_m0_drop,self.alpha) + self.sce_loss(Xkeep, H_m1_keep,self.alpha)
        # The reconstruction loss of node features 
        # features_loss = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(X - X_, 2))))

        # Xdrop = tf.gather(X, drop_indices)
        # X_drop = tf.gather(X_, drop_indices)
        # features_loss = self.sce_loss(X, X_)
        # features_loss0=self.sce_loss(X, X_0)
        weight_decay_loss = 0
        #for layer in range(self.n_layers):    
        #    weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W[layer][0]), self.weight_decay, name='weight_loss_0')        
        #    weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W[layer][1]), self.weight_decay, name='weight_loss_1')
        # Total loss
        weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W[self.n_layers-1][0]), self.weight_decay, name='weight_loss_0')
        weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W[self.n_layers-1][1]), self.weight_decay, name='weight_loss_1')
        self.loss = features_loss + weight_decay_loss
        self.Att_l = self.C
        return self.loss, self.H, self.Att_l, self.H_m0

    def __encoder(self, A, H, layer):
        H1 = tf.matmul(H, self.W[layer][0])
        H2 = tf.matmul(H, self.W[layer][1])
        H = tf.add(H1, H2)
        if layer == self.n_layers - 1:
            return H1
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H1)


    def __decoder(self, H, layer):
        H1 = tf.matmul(H, self.W[layer][0], transpose_b=True)
        # H2 = tf.matmul(H, self.W[layer][1], transpose_b=True)
        # H = tf.add(H1, H2)
        if layer == 0:
            return H1
        return tf.sparse_tensor_dense_matmul(self.C[layer - 1], H1)

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
        learnable_param1 = tf.Variable(tf.zeros((1, hidden_dims[0])), trainable=True,name = "learnable_param1")
        learnable_param2 = tf.Variable(tf.zeros((1, hidden_dims[-1])), trainable=True,name = "learnable_param2")
        return W_d, Ws_att, learnable_param1,learnable_param2

    def graph_attention_layer(self, A, M, v, layer):
        with tf.variable_scope("layer_%s" % layer):
            # f1 = tf.matmul(M, v)

            # f1 = A * f1
            f1 = A * tf.transpose(tf.matmul(tf.nn.sigmoid(M),v),[1, 0])
            unnormalized_attentions1 = tf.SparseTensor(indices=f1.indices,
                                                       values=f1.values,
                                                       dense_shape=f1.dense_shape)
            #unnormalized_attentions2 = tf.sparse_transpose(unnormalized_attentions1, [1, 0])
            #unnormalized_attentions = tf.sparse_add(unnormalized_attentions1, unnormalized_attentions2)
            attentions = tf.sparse_softmax(unnormalized_attentions1)
            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)
            return attentions
