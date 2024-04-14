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

    def __call__(self, A, X, mask_ratio, noise):

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

