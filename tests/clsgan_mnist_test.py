from models.clsgan import CLSGAN
from tensorflow.examples.tutorials.mnist import input_data
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# x_train = (x_train.astype('float32') - 127.5) / 127.5
# x_test = (x_test.astype('float32') - 127.5) / 127.5
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

mnist = input_data.read_data_sets("tmp/MNIST_data/", one_hot=True)
layers = {"g": [64, 64, 64], "d": [128, 128, 128]}
model = CLSGAN(784, 10, 100, layers)


print("Training CLSGAN")
batch_size = 100
epochs = 10000
d_steps = 1
i = 0
for it in tqdm(range(epochs)):
    for _ in range(d_steps):
        X_mb, y_mb = mnist.train.next_batch(batch_size)
        D_loss = model.train_discriminator(X_mb, y_mb)

    X_mb, y_mb = mnist.train.next_batch(batch_size)
    G_loss = model.train_generator(y_mb)
    model.log(X_mb,y_mb,it)
    i += 1

    if i % 500 == 0:
        r, c = 2, 5
        labels = np.arange(0, 10).reshape(-1, 1)
        labels_hot = to_categorical(labels)
        gen_imgs = model.predict(labels_hot)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt].reshape(28, 28))
                axs[i, j].set_title("Digit: %d" % labels[cnt])
                axs[i, j].axis('off')
                cnt += 1
        plt.savefig("img.png")
