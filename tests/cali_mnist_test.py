from models.cali import CALI
from tensorflow.examples.tutorials.mnist import input_data
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("tmp/MNIST_data/", one_hot=True)
layers = {"g": [256], "d": [128, 128]}
model = CALI(784, 10, 100, layers)


print("Training CALI")
batch_size = 100
epochs = 10000
d_steps = 1
i = 0
for it in tqdm(range(epochs)):
    for _ in range(d_steps):
        X_mb, y_mb = mnist.train.next_batch(batch_size)
        D_loss = model.train_discriminator(X_mb, y_mb)

    X_mb, y_mb = mnist.train.next_batch(batch_size)
    G_loss = model.train_generator(X_mb, y_mb)
    model.log(X_mb, y_mb, it)
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
