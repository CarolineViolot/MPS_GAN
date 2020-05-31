#!/usr/bin/env python
# coding: utf-8

# In[31]:


#get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

from tensorflow.keras.optimizers import Adam

import tensorflow as tf


# In[32]:


img_rows = 28
img_cols = 28
channels = 1
img_size = 28

# Input image dimensions
img_shape = (img_rows, img_cols, channels)

# Size of the noise vector, used as input to the Generator
z_dim = 100
alpha = 0.2

# ## Generator

# In[33]:


def build_generator(z_dim):
 #4 deconvolutional layers
 
    model = tf.keras.Sequential()

    # Reshape input into 7x7x256 tensor via a fully connected layer
    
    model.add(Dense(7*7*512, use_bias = False,input_shape=(z_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((7, 7, 512)))
    assert model.output_shape == (None, 7, 7, 512)
    
    
    # Transposed convolution layer, from 7x7x512 into 14x14x256 tensor
    model.add(Conv2DTranspose(256, kernel_size=3, strides=2, padding='same'))
    assert model.output_shape == (None, 14, 14, 256)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    # Transposed convolution layer, from 14x14x256 to 14x14x128 tensor
    model.add(Conv2DTranspose(128, kernel_size=3, strides=1, padding='same'))
    assert model.output_shape == (None, 14, 14, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    # Transposed convolution layer, from 14x14x128 to 14x14x64
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    # Transposed convolution layer, from 14x14x64 to 28x28x1 tensor
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))

    # Output layer with tanh activation
    model.add(Activation('tanh'))

    return model


# ## Discriminator

# In[34]:


def build_discriminator(img_shape):

    model = tf.keras.Sequential()

    # Convolutional layer, from 28x28x1 into 14x14x32 tensor
    model.add(Conv2D(32,kernel_size=3,strides=2,input_shape=img_shape,padding='same'))
    
    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.1))

    # Convolutional layer, from 14x14x32 into 7x7x64 tensor
    model.add(Conv2D(64,kernel_size=3,strides=2,input_shape=img_shape,padding='same'))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.1))

    # Convolutional layer, from 7x7x64 tensor into 3x3x128 tensor
    model.add(Conv2D(128,kernel_size=3,strides=2,input_shape=img_shape,padding='same'))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.1))
    # Output layer with sigmoid activation
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model


# ## Build the Model

# In[35]:


def build_gan(generator, discriminator):

    model = tf.keras.Sequential()

    # Combined Generator -> Discriminator model
    model.add(generator)
    model.add(discriminator)

    return model


# In[36]:


# Build and compile the Discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

# Build the Generator
generator = build_generator(z_dim)

# Keep Discriminator’s parameters constant for Generator training
discriminator.trainable = False

# Build and compile GAN model with fixed Discriminator to train the Generator
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())


# In[37]:


def process_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_png(img, channels = 1, dtype = tf.dtypes.uint8)
    #img = rgb2gray(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    return tf.image.resize(img, [img_size, img_size])

def create_dataset(file_path):
    list_ds = tf.data.Dataset.list_files(file_path)
    img_ds = list_ds.map(process_image)
    for image in img_ds.take(1):
        print("Image shape:", image.numpy().shape)
    img_ds = img_ds.as_numpy_iterator()
    img_ds = np.array(list(img_ds))
    return img_ds

def divide_images(X_train, img_size):
    n_images = math.floor(len(X_train)/img_size)
    new_images = np.array([])
    for x in X_train:
        for i in range(n_images):
            for j in range(n_images):
                new_images(append())


# In[ ]:





# ## Training

# In[52]:


losses = []
accuracies = []
iteration_checkpoints = []


def train(iterations, batch_size, sample_interval):

    # Load the Gaussian dataset
    X_train = create_dataset('Images28x28/Images/*')

    #X_train = X_train.reshape(len(X_train), img_size*img_size)

    # Rescale [0, 255] grayscale pixel values to [-1, 1]
    X_train = X_train / 127.5 - 1.0
    #X_train = np.expand_dims(X_train, axis=3)

    # Labels for real images: all ones
    real = np.ones((batch_size, 1))

    # Labels for fake images: all zeros
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        # -------------------------
        #  Train the Discriminator
        # -------------------------

        # Get a random batch of real images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # Train Discriminator
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train the Generator
        # ---------------------

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # Train Generator
        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % sample_interval == 0:

            # Save losses and accuracies so they can be plotted after training
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            # Output training progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (iteration + 1, d_loss, 100.0 * accuracy, g_loss))

            # Output a sample of generated image
            sample_images(generator, iteration)


# In[59]:


def sample_images(generator,iteration, image_grid_rows=4, image_grid_columns=4 ):

    # Sample random noise
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # Generate images from random noise
    gen_imgs = generator.predict(z)

    # Rescale image pixel values to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Set image grid
    dim = (4, 4)
    plt.figure(figsize=(10,10))
    for i in range(dim[0]*dim[1]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(gen_imgs[i].squeeze(), interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('DCGAN_generated_images/gan_generated_image_epoch_%d.png' % (iteration+1))
    """
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(10, 10),
                            sharey=True,
                            sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output a grid of images
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
            """


# ## Train the Model and Inspect Output

# Note that the `'Discrepancy between trainable weights and collected trainable'` warning from Keras is expected. It is by design: The Generator's trainable parameters are intentionally held constant during Discriminator training, and vice versa.

# In[61]:


# Set hyperparameters
iterations = 25000
batch_size = 128
sample_interval = 1000

# Train the DCGAN for the specified number of iterations
train(iterations, batch_size, sample_interval)


# In[62]:


losses = np.array(losses)

# Plot training losses for Discriminator and Generator
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, losses.T[0], label="Discriminator loss")
plt.plot(iteration_checkpoints, losses.T[1], label="Generator loss")

plt.xticks(iteration_checkpoints, rotation=90)

plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()


# In[63]:


accuracies = np.array(accuracies)

# Plot Discriminator accuracy
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, accuracies, label="Discriminator accuracy")

plt.xticks(iteration_checkpoints, rotation=90)
plt.yticks(range(0, 100, 5))

plt.title("Discriminator Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy (%)")
plt.legend()


# In[ ]:




