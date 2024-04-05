import tensorflow.compat.v1 as tf
import scipy.sparse as sp
import numpy as np
from .model import SMGATE
from tqdm import tqdm

class STMGraph():

    def __init__(self, hidden_dims, n_epochs=500, lr=0.001,alpha=1,
                 gradient_clipping=5, nonlinear=True, weight_decay=0.0001, 
                 verbose=True, random_seed=2023):
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        self.loss_list = []
        self.lr = lr
        self.n_epochs = n_epochs
        self.gradient_clipping = gradient_clipping
        self.build_placeholders()
        self.verbose = verbose
        self.smgate = SMGATE(hidden_dims, nonlinear, weight_decay, alpha)
        self.loss, self.H, self.C, self.ReX = self.smgate(self.A, self.X, self.mask_ratio, self.noise)
        self.optimize(self.loss)
        self.build_session()
        
    def build_placeholders(self):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.mask_ratio = tf.placeholder(dtype=tf.float32)
        self.noise = tf.placeholder(dtype=tf.float32)

    def build_session(self, gpu= True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if gpu == False:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables)) 

    def __call__(self, A, X, mask_ratio, noise):
        for epoch in tqdm(range(self.n_epochs)):
            self.run_epoch(epoch, A, X, mask_ratio, noise)

    def run_epoch(self, epoch, A, X, mask_ratio=0.5,noise=0.1):
        loss, _ = self.session.run([self.loss, self.train_op],
                                         feed_dict={self.A: A,
                                                    self.X: X,
                                                    self.mask_ratio:mask_ratio,
                                                    self.noise:noise})
        self.loss_list.append(loss)
        if self.verbose:
            print("Epoch: %s, Loss: %.4f" % (epoch, loss))
        return loss

    def infer(self, A, X, mask_ratio=0,noise=0):
        H, C, ReX = self.session.run([self.H, self.C, self.ReX],
                           feed_dict={self.A: A,
                                      self.X: X,
                                      self.mask_ratio:mask_ratio,
                                      self.noise:noise})

        return H, self.Conbine_Atten_l(C), self.loss_list, ReX

    def Conbine_Atten_l(self, input):
        return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])), shape=(input[layer][2][0], input[layer][2][1])) for layer in input]

