import numpy as np
import tensorflow as tf
from sklearn.mixture import GaussianMixture


def initialize_trainable_parameters(d_o, d_x, d_h_enc, d_h_dec, K, data):

    #######################
    # parameter for encoder
    #######################

    phi = {}
    # dimensions for all layers
    layer_dim = [d_o]
    layer_dim += d_h_enc
    layer_dim.append(d_x * 2)
    # initialize weights and bias terms of simple feed-forward network
    for l in range(len(layer_dim) - 1):
        # weight matrices
        phi["Enc_W" + str(l)] = tf.get_variable("Enc_W" + str(l),
                                                shape=[layer_dim[l], layer_dim[l + 1]],
                                                initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                                )
        # bias vectors
        phi["Enc_b" + str(l)] = tf.get_variable("Enc_b" + str(l),
                                                shape=[layer_dim[l + 1]],
                                                initializer=tf.constant_initializer(0.0),
                                                )

    #######################
    # parameter for decoder
    #######################

    theta = {}
    # dimensions for all layers
    layer_dim = [d_x]
    layer_dim += d_h_dec
    layer_dim.append(d_o * 2)
    # initialize weights and bias terms for simple feed-forward network
    for l in range(len(layer_dim) - 1):
        # weight matrices
        theta["Dec_W" + str(l)] = tf.get_variable("Dec_W" + str(l),
                                                  shape=[layer_dim[l], layer_dim[l + 1]],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                                  )
        # bias vectors
        theta["Dec_b" + str(l)] = tf.get_variable("Dec_b" + str(l),
                                                  shape=[layer_dim[l + 1]],
                                                  initializer=tf.constant_initializer(0.0),
                                                  )

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    phi = sess.run(phi)
    theta = sess.run(theta)
    tf.reset_default_graph()

    ################
    # SMM parameters
    ################
    xi = {}

    # GMM-based pre-training
    gmm = GaussianMixture(n_components=K).fit(data)
    gamma_gmm = gmm.predict_proba(data)

    """
    xi["pi_k"] = gmm.weights_
    xi["mu_k"] = gmm.means_
    xi["Sigma_k"] = gmm.covariances_
    xi["nu_k"] = 5.0 * np.ones((K, 1))
    """

    # mixing weights
    xi["pi_k"] = np.ones((K, 1))
    # mean vectors
    xi["mu_k"] = np.zeros((K, d_x))
    # covariance matrices
    xi["Sigma_k"] = np.zeros((K, d_x, d_x))
    for k in range(K):
        xi["Sigma_k"][k, :, :] += 0.5 * np.identity(d_x)
    # degrees of freedom
    xi["nu_k"] = 5.0 * np.ones((K, 1))

    return xi, theta, phi, gamma_gmm
