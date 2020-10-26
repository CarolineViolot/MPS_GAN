#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:52:24 2020

@author: caroline
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import time
import os



PATH = os.path.dirname(__file__)
BATCH_SIZE = 128
IMG_ROWS = 100
IMG_COLS = 100
AUTOTUNE = tf.data.experimental.AUTOTUNE
#%%

def process_image(file_path, gray=True):
    img = tf.io.read_file(file_path)
    if gray:
        channels_num = 1
    else : 
        channels_num = 3
    img = tf.io.decode_png(img, channels = channels_num, dtype = tf.dtypes.uint8)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_ROWS, IMG_COLS])

def create_dataset(file_path):
    list_ds = tf.data.Dataset.list_files(file_path)
    img_ds = list_ds.map(process_image)
    for image in img_ds.take(1):
        print("Image shape:", image.numpy().shape)
    return img_ds
    
def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  print('ds shape', len(list(ds)))
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE, drop_remainder=False)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

#X_train = create_dataset(PATH+'/data/images/*')
#%%

def build_generator(img_shape, z_dim):

    model = Sequential()
    
    model.add(Dense(128, input_dim=z_dim)) # Fully connected layer
    model.add(LeakyReLU(alpha=0.01)) # Leaky ReLU activation
    model.add(Dense(img_shape[0] * img_shape[1] * 1, activation='tanh')) # Output layer with tanh activation
    model.add(Reshape(img_shape)) # Reshape the Generator output to image dimensions

    return model

def build_discriminator(img_shape):

    model = Sequential()

    model.add(Flatten(input_shape=img_shape)) # Flatten the input image
    model.add(Dense(128)) # Fully connected layer
    model.add(LeakyReLU(alpha=0.01)) # Leaky ReLU activation
    model.add(Dense(1, activation='sigmoid')) # Output layer with sigmoid activation

    return model

def build_gan(generator, discriminator):

    model = Sequential()

    # Combined Generator -> Discriminator model
    model.add(generator)
    model.add(discriminator)

    return model


# Input image dimensions
print('deal with number of channels here')
img_shape = (IMG_ROWS, IMG_COLS, 1)

# Size of the noise vector, used as input to the Generator
z_dim = 100

# Build and compile the Discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

# Build the Generator
generator = build_generator(img_shape, z_dim)

# Keep Discriminator’s parameters constant for Generator training
discriminator.trainable = False

# Build and compile GAN model with fixed Discriminator to train the Generator
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

losses = []
accuracies = []
iteration_checkpoints = []


def train(iterations, batch_size, sample_interval):
    X_dataset = create_dataset(PATH+'/gray/images/*')
    X_train = prepare_for_training(X_dataset)
    print('type and shape of X_train after prepared for training : ',type(X_dataset))#, X_train.shape)
    # Labels for real images: all ones
    real = np.ones((batch_size, 1))
    # Labels for fake images: all zeros
    fake = np.zeros((batch_size, 1))
    #it = iter(X_train)
    #it = iter(X_train.batch(batch_size, drop_remainder = True))
    for iteration in range(iterations):
        print('iteration :', iteration)
        # -------------------------
        #  Train the Discriminator
        # -------------------------
        #X_train = prepare_for_training(X_dataset)
        #it = iter(X_train)        
        # Get a random batch of real images
        imgs = next(iter(X_train))
        #print(imgs.shape)
        #X_train = next(iter(X_train))
        imgs = imgs / 127.5 - 1.0
        

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        # Train Discriminator
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train the Generator
        # ---------------------

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        # Train Generator
        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % sample_interval == 0:

            # Save losses and accuracies so they can be plotted after training
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)
            print('imgs shape:', imgs.shape)
            # Output training progress
            print("%d [D loss: %f, acc.: %.2f%%] [Gimport time
default_timeit_steps = 1000

def timeit(ds, steps=default_timeit_steps):
  start = time.time()
  it = iter(ds)
  for i in range(steps):
    batch = next(it)
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))
 loss: %f]" %
                  (iteration + 1, d_loss, 100.0 * accuracy, g_loss))

            # Output a sample of generated image
            sample_images(generator)
            
def sample_images(generator, image_grid_rows=4, image_grid_columns=4):

    # Sample random noise
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # Generate images from random noise
    gen_imgs = generator.predict(z)

    # Rescale image pixel values to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Set image grid
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(4, 4),
                            sharey=True,
                            sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output a grid of images
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
            
# Set hyperparameters
iterations = 40

sample_interval = 5

# Train the GAN for the specified number of iterations
train(iterations, BATCH_SIZE, sample_interval)
