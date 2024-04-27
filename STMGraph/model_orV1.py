import tensorflow.compat.v1 as tf

class SGATE():

    def __init__(self,hidden_dims,nonlinear=True, weight_decay=0.0001,alpha=1.0):
        self.n_layers = len(hidden_dims) - 1
        self.alpha = alpha
        self.W, self.v, self.learnable_param1, self.learnable_param2 = self.define_weights(hidden_dims)
        self.C = {}
        self.prune_C = {}
        self.nonlinear = nonlinear
        self.weight_decay = weight_decay

    def sce_loss(self, x, y, alpha=1):
        x = tf.math.l2_normalize(x, axis=-1)
        y = tf.math.l2_normalize(y, axis=-1)

        loss = tf.pow(1 - tf.reduce_sum(tf.multiply(x, y), axis=-1), alpha)
        loss = tf.reduce_mean(loss)
        # loss = tf.reduce_sum(loss)
        return loss

    def __call__(self, A, X, dropout, noise):
        # Encoder
        H = X
        for layer in range(self.n_layers):
            H = self.__encoder(A, H, layer)
            if self.nonlinear:
                if layer != self.n_layers - 1:
                    H = tf.nn.elu(H)
        # Final node representations
        self.H = H
        H_ = H
        # Decoder
        for layer in range(self.n_layers - 1, -1, -1):
            H_ = self.__decoder(H_, layer)
            if self.nonlinear:
                if layer != 0:
                    H_ = tf.nn.elu(H_)
        self.H_ = H_
        # features_loss = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(X - X_, 2))))
        features_loss = self.sce_loss(X, self.H_, self.alpha)
        # H_m2_drop = tf.gather(H_m2, drop_indices)
        
        weight_decay_loss = 0
        # for layer in range(self.n_layers):
        #    weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W[layer][0]), self.weight_decay, name='weight_loss_0')
        #    weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W[layer][1]), self.weight_decay, name='weight_loss_1')
        # Total loss
        weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W[self.n_layers - 1][0]), self.weight_decay,
                                         name='weight_loss_0')
        weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W[self.n_layers - 1][1]), self.weight_decay,
                                         name='weight_loss_1')
        # weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.v[self.n_layers-2]), self.weight_decay, name='weight_v_loss')
        # self.loss = self.alpha*features_loss0 + features_loss + weight_decay_loss
        self.loss = features_loss + weight_decay_loss  # + 0.1*latent_loss
        self.Att_l = self.C
        return self.loss, self.H, self.Att_l, self.H_

    def __encoder(self, A, H, layer):
        H = tf.matmul(H, self.W[layer])
        if layer == self.n_layers - 1:
            return H
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)


    def __decoder(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        if layer == 0:
            return H
        return tf.sparse_tensor_dense_matmul(self.C[layer - 1], H)


    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers):
            W[i] = tf.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))

        Ws_att = {}
        for i in range(self.n_layers - 1):
            v = {}
            v[0] = tf.get_variable("v%s_0" % i, shape=(hidden_dims[i + 1], 1))
            v[1] = tf.get_variable("v%s_1" % i, shape=(hidden_dims[i + 1], 1))

            Ws_att[i] = v
        learnable_param1 = tf.Variable(tf.zeros((1, hidden_dims[0])), trainable=True, name="learnable_param1")
        learnable_param2 = tf.Variable(tf.zeros((1, hidden_dims[-1])), trainable=True, name="learnable_param2")
        return W, Ws_att, learnable_param1, learnable_param2

    def graph_attention_layer(self, A, M, v, layer):

        with tf.variable_scope("layer_%s" % layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)

            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                                      values=tf.nn.sigmoid(logits.values),
                                                      dense_shape=logits.dense_shape)
            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions
