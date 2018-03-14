import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import deepdish as dd


class tVAE:
    """
    Variational Autoencoder with Embedded Student-t Mixture Model
    """
    def __init__(self,
                 d_x,  # latent space dimension
                 d_o,  # dimension of observation space
                 decay_rate,  # decay rate of learning rate
                 decay_step,  # decay step of learning rate
                 num_samples,  # number of training samples
                 d_h_enc,  # list of dimensions of hidden layers for encoder (number of nodes)
                 d_h_dec,  # list of dimensions of hidden layers for decoder (number of nodes)
                 initial_learning_rate,  # starting learning rate
                 sigma_k_2,  # hyper-parameter (variance) for class-specific Gaussian distributions
                 K,  # number of clusters/classes
                 batch_size,  # batch size for all training stages
                 xi_init,  # mixture model parameters for initialization
                 phi_init,  # encoder parameters for initialization
                 theta_init,  # decoder parameters for initialization
                 a_relu,  # parameter for leaky ReLu
                 T,  # number of samples for Gaussian distribution
                 ):

        ############################
        # setup for fixed parameters
        ############################

        # batch size
        self.batch_size = batch_size
        # number of clusters
        self.K = K
        # dimension latent <space
        self.d_x = d_x
        # dimension observation space
        self.d_o = d_o
        # dimension of hidden layers for encoder
        self.d_h_enc = d_h_enc
        # dimension of hidden layers for encoder
        self.d_h_dec = d_h_dec
        # value for leaky ReLu
        self.a_relu = a_relu
        # number of Gaussian samples
        self.T = T
        # for mixture models
        self.sigma_k_2 = sigma_k_2
        # initial learning rate
        self.initial_learning_rate = initial_learning_rate
        # number of training samples for supervised training
        self.num_samples = num_samples
        # decay rate for learning rate
        self.decay_rate = decay_rate
        # decay step-size for learning rate
        self.decay_step = decay_step

        #############################
        # initialize model parameters
        #############################

        # create placeholder for observed variables and posteriors
        self.o, self.gamma = self.create_placeholders()
        # initialize variables for Gaussian mixture model
        self.xi, self.c_k, self.C_k = self.initialize_mixture_model(xi_init)
        # initialize encoder, decoder
        self.phi, self.theta, self.layer_dim_phi, self.layer_dim_theta = self.initialize_encoder_decoder(phi_init,
                                                                                                         theta_init)

        ###########################################
        # encoder + parametrization trick + decoder
        ###########################################

        # encoding
        mu_x, log_sigma_x, sigma_x = self.encoder()

        mu_o = []
        sigma_o = []
        for t in range(self.T):

            # parametrization trick
            epsilon = tf.random_normal(shape=tf.shape(sigma_x))
            x_t = mu_x + tf.multiply(sigma_x, epsilon)

            # decoding
            mu_o_t, sigma_o_t = self.decoder(tf.squeeze(x_t))
            mu_o.append(mu_o_t)
            sigma_o.append(sigma_o_t)

        # list of T tensors to tensor with shape = [B, T, d_x]
        mu_o = tf.stack(mu_o, axis=1)
        sigma_o = tf.stack(sigma_o, axis=1)

        ##################
        # hyper-parameters
        ##################

        # update hyper-parameters
        beta, alpha, log_qz, log_rho = self.update_hyperparameter(mu_x, sigma_x)
        # estimate posteriors
        self.gamma_est = tf.nn.softmax(log_qz)

        ###########################################
        # loss function + learning rate + optimizer
        ###########################################
        # loss
        self.loss, self.neg_log_like, self.neg_entropy, self.neg_cross_entropy \
            = self.compute_loss_function(self.gamma, log_rho, mu_o, sigma_o, sigma_x)
        # learning rate
        self.learning_rate, self.global_step = self.define_learning_rate()
        # optimizer
        self.optimizer = self.define_optimizer()

        #########################################
        # launch session and initialize variables
        #########################################
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def create_placeholders(self):

        # placeholder for observed variables
        o = tf.placeholder(dtype=tf.float32, shape=[None, self.d_o], name="o")
        # placeholder for corresponding labels
        gamma = tf.placeholder(dtype=tf.float32, shape=[None, self.K], name="labels")

        return o, gamma

    def initialize_mixture_model(self, xi_init):

        # dictionary for SMM parameters
        xi = {}

        # weights, shape=[K, 1]
        c_k = tf.Variable(xi_init["pi_k"], trainable=True, dtype=tf.float32, name="pi_k")
        xi["pi_k"] = tf.nn.softmax(c_k, dim=0)

        # mean vectors, shape=[K, d_x]
        xi["mu_k"] = tf.Variable(xi_init["mu_k"], trainable=True, dtype=tf.float32, name="mu_k")

        # covariance matrices, shape=[K, d_x, d_x])
        C_k = tf.Variable(xi_init["Sigma_k"], trainable=True, dtype=tf.float32, name="Sigma_k")

        xi["Sigma_k"] = []
        for k in range(self.K):
            xi["Sigma_k"].append(
                tf.matmul(C_k[k, :, :], C_k[k, :, :], transpose_b=True)
                + tf.multiply(self.sigma_k_2, tf.eye(self.d_x, dtype=tf.float32)))
        xi["Sigma_k"] = tf.convert_to_tensor(xi["Sigma_k"])

        # degrees of freedom
        xi["nu_k"] = tf.Variable(xi_init["nu_k"], trainable=True, dtype=tf.float32, name="nu_k")

        return xi, c_k, C_k

    def initialize_encoder_decoder(self, phi_init, theta_init):

        #########
        # encoder
        #########

        # parameter dictionary
        phi = {}
        # dimensions for all layers
        layer_dim_phi = [self.d_o]
        layer_dim_phi += self.d_h_enc
        layer_dim_phi.append(self.d_x * 2)
        # initialize weights and bias terms of simple feed-forward network
        for l in range(len(layer_dim_phi) - 1):
            # weight matrices
            phi["Enc_W" + str(l)] = tf.Variable(phi_init["Enc_W" + str(l)],
                                                trainable=True,
                                                dtype=tf.float32,
                                                name="Enc_W" + str(l)
                                                )
            # bias vectors
            phi["Enc_b" + str(l)] = tf.Variable(phi_init["Enc_b" + str(l)],
                                                trainable=True,
                                                dtype=tf.float32,
                                                name="Enc_b" + str(l)
                                                )
        #########
        # decoder
        #########

        # parameter dictionary
        theta = {}
        # dimensions for all layers
        layer_dim_theta = [self.d_x]
        layer_dim_theta += self.d_h_dec
        layer_dim_theta.append(self.d_o * 2)
        # initialize weights and bias terms for simple feed-forward network
        for l in range(len(layer_dim_theta) - 1):
            # weight matrices
            theta["Dec_W" + str(l)] = tf.Variable(theta_init["Dec_W" + str(l)],
                                                  trainable=True,
                                                  dtype=tf.float32,
                                                  name="Dec_W" + str(l)
                                                  )
            # bias vectors
            theta["Dec_b" + str(l)] = tf.Variable(theta_init["Dec_b" + str(l)],
                                                  trainable=True,
                                                  dtype=tf.float32,
                                                  name="Dec_b" + str(l)
                                                  )

        return phi, theta, layer_dim_phi, layer_dim_theta

    def encoder(self):

        # compute forward step
        h = self.o
        for l in range(len(self.layer_dim_phi) - 2):
            # nonlinear function, tf.nn.relu
            h = self.LeakyReLU(tf.add(tf.matmul(h, self.phi["Enc_W" + str(l)]), self.phi["Enc_b" + str(l)]),
                               self.a_relu)

        # finally, linear output
        h = tf.add(tf.matmul(h, self.phi["Enc_W" + str(len(self.layer_dim_phi) - 2)]),
                   self.phi["Enc_b" + str(len(self.layer_dim_phi) - 2)])

        # extract mean and (diagonal) log variance of latent variable
        mu_x = h[:, :self.d_x]
        log_sigma_x = h[:, self.d_x:]

        # calculate standard deviation for practical use
        sigma_x = tf.exp(log_sigma_x)

        return mu_x, log_sigma_x, sigma_x

    def decoder(self, x):

        # compute forward step
        h = x
        for l in range(len(self.layer_dim_theta) - 2):
            # nonlinear function, tf.nn.relu
            h = self.LeakyReLU(tf.add(tf.matmul(h, self.theta["Dec_W" + str(l)]), self.theta["Dec_b" + str(l)]),
                               self.a_relu)
        # final linear output
        h = tf.add(tf.matmul(h, self.theta["Dec_W" + str(len(self.layer_dim_theta) - 2)]),
                   self.theta["Dec_b" + str(len(self.layer_dim_theta) - 2)])

        # extract mean
        mu_o = h[:, :self.d_o]
        # extract standard derivation
        log_sigma_o = h[:, self.d_o:]
        sigma_o = tf.exp(log_sigma_o)

        return mu_o, sigma_o

    def update_hyperparameter(self, mu_x, sigma_x):

        # epsilon value for log computations
        eps = 1e-8

        ###############
        # compute alpha
        ###############
        # size = [K, 1]
        alpha = 0.5 * tf.add(self.xi["nu_k"], self.d_x)

        ################################
        # compute beta and log q(z_nk=1)
        ################################
        beta = []
        log_qz = []

        for k in range(self.K):
            # size = [d_x, d_x]
            Sigma_k = self.xi["Sigma_k"][k, :, :]
            inv_Sigma_k = tf.matrix_inverse(Sigma_k)
            # size = ()
            det_Sigma_k = tf.matrix_determinant(Sigma_k)
            log_det_Sigma_k = tf.log(det_Sigma_k + eps)

            ##############################
            # compute Mahalanobis distance
            ##############################

            # batch-wise subtraction of mean vectors, size = [?, self.d_x]
            sub_mu_x_mu_k = tf.subtract(mu_x, self.xi["mu_k"][k, :])

            # matrix multiplication of the form X A X^T, size = [?, ?]
            x_mu_Sigma_mu_x = tf.matmul(tf.matmul(sub_mu_x_mu_k, inv_Sigma_k), sub_mu_x_mu_k, transpose_b=True)
            # size = (?,)
            x_mu_Sigma_mu_x = tf.diag_part(x_mu_Sigma_mu_x)

            ###############
            # compute trace
            ###############

            # batch-wise multiplication of all diagonal matrices with inverse of Sigma_k
            sigma_x_2 = tf.matrix_diag(tf.square(sigma_x))
            sigma_x_2 = tf.reshape(sigma_x_2, shape=[-1, self.d_x])
            sigma_x_2_inv_sigma_k = tf.matmul(sigma_x_2, inv_Sigma_k)
            # reshape
            sigma_x_2_inv_sigma_k = tf.reshape(sigma_x_2_inv_sigma_k, [-1, self.d_x, self.d_x])
            # compute trace, size = (?,)
            tr_sigma_x_2_inv_sigma_k = tf.trace(sigma_x_2_inv_sigma_k)

            ##############
            # compute beta
            ##############
            # for single class: size = (?,)
            curr_beta = 0.5 * (self.xi["nu_k"][k, :] + x_mu_Sigma_mu_x + tr_sigma_x_2_inv_sigma_k)
            beta.append(curr_beta)

            #########################
            # compute log( q(z_nk=1 )
            #########################
            log_qz.append(tf.log(self.xi["pi_k"][k] + eps)
                          + 0.5 * self.xi["nu_k"][k, :] * tf.log(0.5 * self.xi["nu_k"][k, :] + eps)
                          - tf.lgamma(0.5 * self.xi["nu_k"][k, :] + eps)
                          - 0.5 * log_det_Sigma_k
                          + tf.lgamma(alpha[k] + eps)
                          - alpha[k] * tf.log(curr_beta + eps)
                          )

        # from list tensor with size = [?, K]
        beta = tf.transpose(tf.convert_to_tensor(beta))
        # from list tensor with size = [?, K]
        log_qz = tf.transpose(tf.convert_to_tensor(log_qz))

        #################
        # compute log rho
        #################
        alpha_digamma = tf.reshape((alpha - 1) * tf.digamma(alpha + eps), [1, self.K])
        log_gamma_alpha = tf.reshape(tf.lgamma(alpha + eps), [1, self.K])
        log_rho = log_qz + alpha_digamma + tf.log(beta + eps) - tf.reshape(alpha, [1, self.K]) \
                  - 0.5 * self.d_x * tf.log(2 * np.pi) - log_gamma_alpha

        return beta, alpha, log_qz, log_rho

    def compute_loss_function(self, gamma, log_rho, mu_o, sigma_o, sigma_x):

        ##################################
        # calculate negative cross entropy
        ##################################
        neg_cross_entropy = self.compute_neg_cross_entropy(gamma, log_rho)
        neg_cross_entropy /= self.batch_size

        ############################################
        # calculate negative Gaussian log-likelihood
        ############################################
        neg_log_like = 0.0
        for t in range(self.T):
            mu_o_t = tf.squeeze(mu_o[:, t, :])
            sigma_o_2_t = tf.squeeze(tf.square(sigma_o[:, t, :]))
            neg_log_like += self.compute_neg_gaussian_log_likelihood(self.o, self.d_o, mu_o_t, sigma_o_2_t)/self.T
        neg_log_like /= self.batch_size

        ############################
        # calculate negative entropy
        ############################
        neg_entropy = self.compute_neg_entropy(sigma_x, self.d_x)
        neg_entropy /= self.batch_size

        ###################
        # combine all terms
        ###################
        loss = neg_log_like + neg_entropy + neg_cross_entropy

        return loss, neg_log_like, neg_entropy, neg_cross_entropy

    def define_learning_rate(self):

        #######################
        # decayed learning rate
        #######################
        global_step = tf.Variable(0, dtype=tf.float32, trainable=False)
        num_batches_per_epoch = int(self.num_samples / float(self.batch_size))
        decay_steps = int(num_batches_per_epoch * self.decay_step)
        learning_rate = tf.train.exponential_decay(learning_rate=self.initial_learning_rate,  # start learning rate
                                                   global_step=global_step,  # counter
                                                   decay_steps=decay_steps,  # update after 'decay_step' epochs
                                                   decay_rate=self.decay_rate,  # decay rate
                                                   staircase=False
                                                   )
        return learning_rate, global_step

    def define_optimizer(self):

        adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads_and_vars = adam.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_norm(grad, 1.0), var) for grad, var in grads_and_vars]
        optimizer = adam.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)

        return optimizer

    @staticmethod
    def compute_neg_cross_entropy(gamma, log_rho):

        # cross entropy
        cross_entropy = -tf.reduce_sum(tf.multiply(gamma, log_rho))

        return cross_entropy

    @staticmethod
    def compute_neg_entropy(sigma_x, d_x, eps=1e-8):

        # compute negative entropy for N(x| mu_x, sigma_x)
        sigma_x_2 = tf.square(sigma_x)
        entropy = -0.5 * tf.reduce_sum(d_x * tf.log(2 * np.pi)
                                       + tf.reduce_sum(tf.log(sigma_x_2 + eps), axis=1, keep_dims=True)
                                       + d_x
                                       )
        return entropy

    @staticmethod
    def compute_neg_gaussian_log_likelihood(o, d_o, mu_o, sigma_o, eps=1e-8):

        # o - mu_o, size = [?, d_o]
        omu2 = tf.square(tf.subtract(o, mu_o))
        # (o-mu_o)^T Inv_Sigma_o (o-mu_o), size = [?, d_o]
        omu_sigma_omu = tf.reduce_sum(tf.multiply(omu2, tf.reciprocal(sigma_o + eps)), axis=1)

        # compute negative log likelihood for observation
        log_like = 0.5 * tf.reduce_sum(d_o * tf.log(2*np.pi)
                                       + tf.reduce_sum(tf.log(sigma_o + eps), axis=1)
                                       + omu_sigma_omu
                                       )
        return log_like

    def update_supervised(self, o, labels):

        # update model variables and compute loss
        _, loss, loglike, entropy, cross_entropy = self.sess.run([self.optimizer,
                                                                  self.loss,
                                                                  self.neg_log_like,
                                                                  self.neg_entropy,
                                                                  self.neg_cross_entropy],
                                                                 feed_dict={self.o: o, self.gamma: labels}
                                                                 )
        return loss, loglike, entropy, cross_entropy

    def update_unsupervised(self, o):

        # estimate labels
        gamma = self.sess.run(self.gamma_est, feed_dict={self.o: o})

        # update model variables and compute loss
        _, loss, loglike, entropy, cross_entropy = self.sess.run([self.optimizer,
                                                                  self.loss,
                                                                  self.neg_log_like,
                                                                  self.neg_entropy,
                                                                  self.neg_cross_entropy],
                                                                 feed_dict={self.o: o, self.gamma: gamma}
                                                                 )
        return loss, loglike, entropy, cross_entropy

    def get_hyper_parameters(self):

        # apply decoder for input observation o
        mu_k, Sigma_k, nu_k = self.sess.run([self.xi["mu_k"],
                                             self.xi["Sigma_k"],
                                             self.xi["nu_k"]]
                                            )
        return mu_k, Sigma_k, nu_k

    def decode(self, x):
        mu_o, sigma_o = self.sess.run(self.decoder(np.asarray(x, dtype=np.float32)))
        return mu_o, sigma_o

    def get_learning_rate(self):

        # get current learning rate
        learning_rate = self.sess.run(self.learning_rate)

        return learning_rate

    @staticmethod
    def LeakyReLU(x, a_relu):
        return tf.nn.relu(x) - a_relu * tf.nn.relu(-x)

    def plot_results(self, data, path, epoch):

        fig = plt.figure(figsize=(15, 10))

        ###############################
        # subplot 1: plot training data
        ###############################
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.scatter(data[:, 0], data[:, 1], marker=".", linewidths=0.2)
        ax1.set_title('training samples')
        ax1.set_xlim((-1.5, 1.5))
        ax1.set_ylim((-1.5, 1.5))

        ###############################################
        # subplot 2: plot x_n ~ N(mu_k, Sigma_k / u_nk)
        ###############################################
        # all hyper-parameters
        mu_k, Sigma_k, nu_k = self.get_hyper_parameters()
        # labels
        labels = np.asarray(1000 * [0] + 1000 * [1] + 1000 * [2] + 1000 * [3] + 1000 * [4])
        # number of training samples
        num_samples = len(labels)
        # sample u_nk and x_n values
        u = np.zeros((num_samples, 1))
        x = np.zeros((num_samples, self.d_x))
        for n in range(num_samples):
            k = labels[n]
            u[n] = np.random.gamma(shape=nu_k[k] / 2, scale=2 / nu_k[k])
            x[n, :] = np.random.multivariate_normal(mean=mu_k[k, :], cov=Sigma_k[k, :, :] / u[n])
        # plot x_n
        ax2 = fig.add_subplot(2, 2, 2)
        for k in range(self.K):
            ax2.scatter(x[labels == k, 0], x[labels == k, 1], marker=".", linewidths=0.2)
        ax2.set_title('x_n ~ N(mu_k, Sigma_k/u_nk)')
        ax2.set_xlim((-1.5, 1.5))
        ax2.set_ylim((-1.5, 1.5))

        ######################
        # subplot 3: plot mu_o
        ######################
        mu_o, sigma_o = self.decode(x)
        ax3 = fig.add_subplot(2, 2, 3)
        for i in range(self.K):
            ax3.scatter(mu_o[labels == i, 0], mu_o[labels == i, 1], marker=".", linewidths=0.2)
        ax3.set_title('mu_o, {mu_o, sigma_o} = dec(x)')
        ax3.set_xlim((-1.5, 1.5))
        ax3.set_ylim((-1.5, 1.5))

        #######################
        # subplot 4: plot o_new
        #######################
        ax4 = fig.add_subplot(2, 2, 4)
        o = np.zeros((num_samples, self.d_o))
        for n in range(num_samples):
            o[n, :] = np.random.multivariate_normal(mean=mu_o[n, :], cov=np.diag(np.square(sigma_o[n, :])))
        for i in range(self.K):
            ax4.scatter(o[labels == i, 0], o[labels == i, 1], marker=".", linewidths=0.2)
        ax4.set_title('o ~ N(mu_o, diag(sigma_o))')
        ax4.set_xlim((-1.5, 1.5))
        ax4.set_ylim((-1.5, 1.5))

        ###########
        # save file
        ###########
        path_file = os.path.join(path, epoch)
        plt.savefig('{}.png'.format(path_file), bbox_inches='tight', pad_inches=0)
        plt.close()

    def save_parameters(self, counter, learning_rate_curve, dof_curve):

        parameter_dic = {"theta": self.sess.run(self.theta),
                         "phi": self.sess.run(self.phi),
                         "xi": self.sess.run(self.xi),
                         "d_x": self.d_x,
                         "d_o":self.d_o,
                         "decay_rate": self.decay_rate,
                         "decay_step": self.decay_step,
                         "num_samples": self.num_samples,
                         "d_h_enc": self.d_h_enc,
                         "d_h_dec": self.d_h_dec,
                         "initial_learning_rate": self.initial_learning_rate,
                         "sigma_k_2": self.sigma_k_2,
                         "K": self.K,
                         "batch_size": self.batch_size,
                         "a_relu": self.a_relu,
                         "dof_curve": dof_curve,
                         "learning_rate_curve": learning_rate_curve,
                         "T": self.T,
                         }
        file = os.path.join("data", "model", "parameter_epoch_" + str(counter) + ".h5")
        dd.io.save(file, parameter_dic)
