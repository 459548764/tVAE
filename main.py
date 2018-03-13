import numpy as np
from tqdm import tqdm
import tensorflow as tf
from model import tVAE
import torchfile
import os
from initialize_parameters import initialize_trainable_parameters

#########################
# folder to store results
#########################
path_pics = os.path.join(os.path.join("data", "pics"))
path_model = os.path.join(os.path.join("data", "model"))
if not os.path.exists(path_pics):
    os.makedirs(path_pics)
if not os.path.exists(path_model):
    os.makedirs(path_model)

#############
# reset graph
#############
tf.reset_default_graph()

#####################
# load synthetic data
#####################
print("load data...")
data = torchfile.load(os.path.join("spiral.t7"))

###################
# define parameters
###################
print("initialize parameters and model...")

# number of clusters
K = 5
# latent dimension
d_x = 2
# observation dimension
d_o = 2
# hidden nodes
d_h_enc = [512, 512]
d_h_dec = [512, 512]
# batch size
batch_size = 100
# number of samples
num_samples = data.shape[0]

# number of epochs
epochs = 100  # 3000
# number of (mini batch) updates performed per epoch
num_batches = int(num_samples / batch_size)
# how much to decay the learning rate
decay_rate = 0.01
# decay step for learning rates
decay_step = epochs

##############
# initializing
##############
xi_init, theta_init, phi_init, gamma_gmm = initialize_trainable_parameters(d_o, d_x, d_h_enc, d_h_dec, K, data)

##############
# define model
##############
tvae = tVAE(d_x=d_x,
            d_o=d_o,
            decay_rate=decay_rate,
            decay_step=decay_step,
            num_samples=num_samples,
            sigma_k_2=0.001,
            K=K,
            batch_size=batch_size,
            d_h_enc=[512, 512],
            d_h_dec=[512, 512],
            initial_learning_rate=2e-4,
            theta_init=theta_init,
            phi_init=phi_init,
            xi_init=xi_init,
            a_relu=0.001,
            T=1,
            )

###########################
# launch training procedure
###########################
print("start training...")
learning_rate_curve = np.zeros((epochs,), dtype='float32')
dof_curve = np.zeros((epochs, K), dtype='float32')

# print current results
tbar = tqdm(range(epochs))

# start training
for epoch in tbar:

    # epoch-wise results
    loss, neg_entropy, neg_log_like, neg_cross_entropy = 0.0, 0.0, 0.0, 0.0

    # learning rate (at the beginning of the epoch)
    learning_rate_curve[epoch] = tvae.get_learning_rate()
    # dof curve
    dof_curve[epoch, :] = tvae.sess.run(tvae.xi["nu_k"]).reshape(-1)

    # shuffle training data
    idx = np.random.permutation(num_samples)
    data = data[idx, :]
    # shuffle gmm posteriors
    gamma_gmm = gamma_gmm[idx, :]

    # make batches
    batches_data = np.array_split(data, num_batches, axis=0)
    batches_labels = np.array_split(gamma_gmm, num_batches, axis=0)

    # mini-batch updates
    for batch_data, batch_labels in zip(batches_data, batches_labels):

        if epoch < 15:
            # pre-training
            curr_loss, curr_neg_log_like, curr_neg_entropy, curr_neg_cross_entropy \
                = tvae.update_supervised(batch_data, batch_labels)
        else:
            # unsupervised training
            curr_loss, curr_neg_log_like, curr_neg_entropy, curr_neg_cross_entropy \
                = tvae.update_unsupervised(batch_data)

        # update results
        neg_log_like += curr_neg_log_like
        neg_entropy += curr_neg_entropy
        loss += curr_loss
        neg_cross_entropy += curr_neg_cross_entropy

    # normalize results after the update of all mini-batches
    loss /= num_batches
    neg_entropy /= num_batches
    neg_log_like /= num_batches
    neg_cross_entropy /= num_batches

    # update progress bar
    s = "Loss: {:.4f}, " \
        "NegEntropy: {:.4f}, " \
        "NegLogLik: {:.4f}, " \
        "NegCrossEnt: {:.4f}, "\
        "LearnRate: {:.4f}".format(
         loss,
         neg_entropy,
         neg_log_like,
         neg_cross_entropy,
         learning_rate_curve[epoch]
    )
    tbar.set_description(s)

    #############################
    # update plots and save model
    #############################
    """
    if epoch % 5 == 0:

        # plots
        tvae.plot_results(data, path_pics, str(epoch))
        # save model
        #tvae.save_parameters(epoch, learning_rate_curve, dof_curve)
    """

writer = tf.summary.FileWriter("summary", tvae.sess.graph)

