# Code for "tVAE: Variational Autoencoder with Embedded Student-t Mixture Model"
under review, paper submitted for publication

# Abstract
Autoencoders (AE) have had tremendous success in learning latent representations of unlabeled data. With 'classical' autoencoders, however, it is difficult to exploit insights into the statistical properties of the latent representations. Variational autoencoders (VAE), on the other hand, provide not only point estimates of the means, but also estimates of the variances, both in latent space and with respect to the observation space. While existing VAE paradigms prove to be efficient and useful in various unsupervised learning contexts, they are currently still bound by limitations imposed by the assumed Gaussianity of all underlying probability distributions. In this work we are extending the Gaussian model for the VAE to a Student-t model, which allows for an independent control of the "heaviness" of the respective tails of the implied probability densities. Experiments over the MNIST dataset show that the proposed method provides superior clustering accuracies to comparable Gaussian-based VAEs.

# Instructions
* Clustering synthetic spiral dataset:
```shell 
python main.py
```
* Clustering MNIST dataset: comming soon
