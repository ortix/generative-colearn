import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
mb_size = 32
z_dim = 10
eps_dim = 4
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-2
beta_1 = 0.5


def log(x):
    return tf.log(x + 1e-8)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


""" Q(z|X,eps) """
X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])
eps = tf.placeholder(tf.float32, shape=[None, eps_dim])

Q_W1 = tf.Variable(xavier_init([X_dim + y_dim + eps_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
Q_W2 = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2 = tf.Variable(tf.zeros(shape=[z_dim]))

theta_Q = [Q_W1, Q_W2, Q_b1, Q_b2]


def Q(X, y, eps):
    inputs = tf.concat(axis=1, values=[X, y, eps])
    h = tf.nn.relu(tf.matmul(inputs, Q_W1) + Q_b1)
    z = tf.matmul(h, Q_W2) + Q_b2
    return z


""" P(X|z) """
P_W1 = tf.Variable(xavier_init([z_dim+y_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_P = [P_W1, P_W2, P_b1, P_b2]


def P(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    h = tf.nn.relu(tf.matmul(inputs, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits


""" D(z) """
D_W1 = tf.Variable(xavier_init([X_dim + z_dim + y_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


def D(X, z, y):
    inputs = tf.concat([X, z, y], axis=1)
    h = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    return tf.matmul(h, D_W2) + D_b2


""" Training """
z_sample = Q(X, y, eps)
_, X_logits = P(z_sample, y)
D_sample = D(X, z_sample, y)

D_q = tf.nn.sigmoid(D(X, z_sample, y))
D_prior = tf.nn.sigmoid(D(X, z, y))

# Sample from random z
X_samples, _ = P(z, y)

disc = tf.reduce_mean(-D_sample)
nll = tf.reduce_sum(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=X_logits, labels=X),
    axis=1
)
loglike = -tf.reduce_mean(nll)

elbo = disc + loglike
D_loss = tf.reduce_mean(log(D_q) + log(1. - D_prior))

VAE_solver = tf.train.AdamOptimizer().minimize(-elbo, var_list=theta_P+theta_Q)
D_solver = tf.train.AdamOptimizer().minimize(-D_loss, var_list=theta_D)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    X_mb, y_mb = mnist.train.next_batch(mb_size)
    eps_mb = np.random.randn(mb_size, eps_dim)
    z_mb = np.random.randn(mb_size, z_dim)

    _, elbo_curr = sess.run([VAE_solver, elbo],
                            feed_dict={X: X_mb, eps: eps_mb, z: z_mb, y: y_mb})

    _, D_loss_curr = sess.run([D_solver, D_loss],
                              feed_dict={X: X_mb, eps: eps_mb, z: z_mb, y: y_mb})

    if it % 1000 == 0:
        print('Iter: {}; ELBO: {:.4}; D_Loss: {:.4}'
              .format(it, elbo_curr, D_loss_curr))

        samples = sess.run(X_samples, feed_dict={z: np.random.randn(10, z_dim), y: np.eye(10)})

        fig = plot(samples)
        plt.savefig('tmp/out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
