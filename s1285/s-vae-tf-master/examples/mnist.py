
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/', one_hot=False)


class ModelVAE(object):

    def __init__(self, x, h_dim, z_dim, activation=tf.nn.relu, distribution='normal'):
        """
        ModelVAE initializer

        :param x: placeholder for input
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        self.x, self.h_dim, self.z_dim, self.activation, self.distribution = x, h_dim, z_dim, activation, distribution

        self.z_mean, self.z_var = self._encoder(self.x)

        if distribution == 'normal':
            self.q_z = tf.distributions.Normal(self.z_mean, self.z_var)
        elif distribution == 'vmf':
            self.q_z = VonMisesFisher(self.z_mean, self.z_var)
        else:
            raise NotImplemented


        self.z = self.q_z.sample()

        self.logits = self._decoder(self.z)

    def _encoder(self, x):
        """
        Encoder network

        :param x: placeholder for input
        :return: tuple `(z_mean, z_var)` with mean and concentration around the mean
        """
        
        # 2 hidden layers encoder
        h0 = tf.layers.dense(x, units=self.h_dim * 2, activation=self.activation)
        h1 = tf.layers.dense(h0, units=self.h_dim, activation=self.activation)

        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            z_mean = tf.layers.dense(h1, units=self.z_dim, activation=None)
            z_var = tf.layers.dense(h1, units=self.z_dim, activation=tf.nn.softplus)
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            z_mean = tf.layers.dense(h1, units=self.z_dim, activation=lambda x: tf.nn.l2_normalize(x, axis=-1))
            # the `+ 1` prevent collapsing behaviors
            z_var = tf.layers.dense(h1, units=1, activation=tf.nn.softplus) + 1
        else:
            raise NotImplemented

        return z_mean, z_var

    def _decoder(self, z):
        """
        Decoder network

        :param z: tensor, latent representation of input (x)
        :return: logits, `reconstruction = sigmoid(logits)`
        """
        # 2 hidden layers decoder
        h2 = tf.layers.dense(z, units=self.h_dim, activation=self.activation)
        h2 = tf.layers.dense(h2, units=self.h_dim * 2, activation=self.activation)
        logits = tf.layers.dense(h2, units=self.x.shape[-1], activation=None)

        return logits


class OptimizerVAE(object):

    def __init__(self, model, learning_rate=1e-3):
        """
        OptimizerVAE initializer

        :param model: a model object
        :param learning_rate: float, learning rate of the optimizer
        """

        # binary cross entropy error
        self.bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=model.x, logits=model.logits)
        self.reconstruction_loss = tf.reduce_mean(tf.reduce_sum(self.bce, axis=-1))

        if model.distribution == 'normal':
            # KL divergence between normal approximate posterior and standard normal prior
            self.p_z = tf.distributions.Normal(tf.zeros_like(model.z), tf.ones_like(model.z))
            kl = model.q_z.kl_divergence(self.p_z)
            self.kl = tf.reduce_mean(tf.reduce_sum(kl, axis=-1))
        elif model.distribution == 'vmf':
            # KL divergence between vMF approximate posterior and uniform hyper-spherical prior
            self.p_z = HypersphericalUniform(model.z_dim - 1, dtype=model.x.dtype)
            kl = model.q_z.kl_divergence(self.p_z)
            self.kl = tf.reduce_mean(kl)
        else:
            raise NotImplemented

        self.ELBO = - self.reconstruction_loss - self.kl

        self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-self.ELBO)

        self.print = {'recon loss': self.reconstruction_loss, 'ELBO': self.ELBO, 'KL': self.kl}


def log_likelihood(model, optimizer, n=10):
    """

    :param model: model object
    :param optimizer: optimizer object
    :param n: number of MC samples
    :return: MC estimate of log-likelihood
    """

    z = model.q_z.sample(n)

    log_p_z = optimizer.p_z.log_prob(z)

    if model.distribution == 'normal':
        log_p_z = tf.reduce_sum(log_p_z, axis=-1)

    log_p_x_z = -tf.reduce_sum(optimizer.bce, axis=-1)

    log_q_z_x = model.q_z.log_prob(z)

    if model.distribution == 'normal':
        log_q_z_x = tf.reduce_sum(log_q_z_x, axis=-1)

    return tf.reduce_mean(tf.reduce_logsumexp(
        tf.transpose(log_p_x_z + log_p_z - log_q_z_x) - np.log(n), axis=-1))

# hidden dimension and dimension of latent space
H_DIM = 500
Z_DIM = 20

# digit placeholder
x = tf.placeholder(tf.float32, shape=(None, 784))

# normal VAE
modelN = ModelVAE(x=x, h_dim=H_DIM, z_dim=Z_DIM, distribution='normal')
optimizerN = OptimizerVAE(modelN)

# hyper-spherical VAE
modelS = ModelVAE(x=x, h_dim=H_DIM, z_dim=Z_DIM + 1, distribution='vmf')
optimizerS = OptimizerVAE(modelS)

session = tf.Session()
session.run(tf.global_variables_initializer())

#print('##### Normal VAE #####')
#for i in range(1000):
#    # training
#    x_mb, _ = mnist.train.next_batch(100)
#    # dynamic binarization
#    x_mb = (x_mb > np.random.random(size=x_mb.shape)).astype(np.float32)
#    
#    session.run(optimizerN.train_step, {modelN.x: x_mb})
#
#    # every 100 iteration plot validation
#    if i % 100 == 0:
#        x_mb = mnist.validation.images
#        # dynamic binarization
#        x_mb = (x_mb > np.random.random(size=x_mb.shape)).astype(np.float32)
#        True
#        print(i, session.run({**optimizerN.print}, {modelN.x: x_mb}))
#
#print('Test set:')
##x_mb = mnist.test.images
#x_mb, y_mb = mnist.train.next_batch(1000)
## dynamic binarization
##x_mb = (x_mb > np.random.random(size=x_mb.shape)).astype(np.float32)
#    
#print_ = {**optimizerN.print}
#print_['LL'] = log_likelihood(modelN, optimizerN, n=100)
#print(session.run(print_, {modelN.x: x_mb}))
#x_latent=session.run(modelN.z_mean, {modelS.x: x_mb}).astype(np.float32)
#result=KMeans(n_clusters=10,random_state=100).fit(x_latent).labels_
#
#true=0
#num=0
#x=[]
#for i in range (10):
#    b=[j for j,v in enumerate(result) if v==i]
#    for k in b:
#        x.append((np.argwhere(y_mb[k,:]==1)))
#    x=list(map(int,x))
#    real= max(x,key =x.count)
#    for m in b:
#        result[m]=real
#    x=[]
#for i in range (np.size(result)):
#    if y_mb[i,result[i]]==1:
#        true+=1
#print(true/np.size(result))
#print()
print('##### Hyper-spherical VAE #####')
for i in range(1000):
    # training
    x_mb, _ = mnist.train.next_batch(100)
    # dynamic binarization
#    x_mb = (x_mb > np.random.random(size=x_mb.shape)).astype(np.float32)
    
    session.run(optimizerS.train_step, {modelS.x: x_mb})

    # every 100 iteration plot validation
    if i % 100 == 0:
        x_mb = mnist.validation.images
        # dynamic binarization
        x_mb = (x_mb > np.random.random(size=x_mb.shape)).astype(np.float32)
        
        print(i, session.run({**optimizerS.print}, {modelS.x: x_mb}))

print('Test set:')
#x_mb = mnist.test.images
x_mb, y_mb = mnist.train.next_batch(1000)
# dynamic binarization
#x_mb = (x_mb > np.random.random(size=x_mb.shape)).astype(np.float32)
    
print_ = {**optimizerS.print}
print_['LL'] = log_likelihood(modelS, optimizerS, n=100)
print(session.run(print_, {modelS.x: x_mb}))

x_latent=session.run(modelS.z_mean, {modelS.x: x_mb}).astype(np.float32)
result=KMeans(n_clusters=10,random_state=100).fit_predict(x_latent)
print(accuracy_score(y_mb,result))

#true=0
#num=0
#x=[]
#for i in range (10):
#    b=[j for j,v in enumerate(result) if v==i]
#    for k in b:
#        x.append((np.argwhere(y_mb[k,:]==1)))
#    x=list(map(int,x))
#    real= max(x,key =x.count)
#    for m in b:
#        result[m]=real
#    x=[]
#for i in range (np.size(result)):
#    if y_mb[i,result[i]]==1:
#        true+=1
#print(true/np.size(result))

#plt.figure(figsize=(8, 6))
#plt.subplot(121)
#plt.scatter(x_latent[:,0],x_latent[:,1], c=result)
#plt.grid()
#plt.subplot(122)
#plt.scatter(x_latent[:,0],x_latent[:,1], c=np.argmax(y_mb, 1))
#plt.colorbar()
#plt.grid()
#plt.show()
#Tf_png='test3.png'
#plt.savefig(Tf_png,dpi=150)

