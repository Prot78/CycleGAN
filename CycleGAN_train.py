#pip3 install -q git+https://github.com/tensorflow/examples.git

##  INPUT informatin   ##########################################
EPOCHS = 50
train_data_path_A = 'Train_data_T1mouse/images/'
train_data_path_B = 'Train_data_T2mouse/images/'

IMG_CHANNELS = 3

###############################################################

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

import time
from IPython.display import clear_output
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imshow, imsave
#from skimage.transform import resize
import matplotlib.pyplot as plt
import png, os, shutil
import pydicom
from datetime import datetime

seed = 42
np.random.seed = seed
AUTOTUNE = tf.data.AUTOTUNE




# TRAIN_PATH


train_list_A  = os.listdir(train_data_path_A)
train_list_B  = os.listdir(train_data_path_B)

#####################################################################


###########################################################
for filename in sorted(os.listdir(train_data_path_A)):
    if filename == '.DS_Store' :
        path = '.DS_Store'
        os.unlink(path)
        os.remove(path)       ### Insert this if needed!!!!!!!!!
for filename in sorted(os.listdir(train_data_path_B)):
    if filename == '.DS_Store':
        path = '.DS_Store'
        os.unlink(path)
        os.remove(path)       ### Insert this if needed!!!!!!!!!
###########################################################




#############  Calculating the lenght of X_train and Y_train  #########################

lenght = 0
for i, image_name in enumerate(sorted(train_list_A)):
    image_name = image_name.strip("._")     #This is for ERISXdl
    if (image_name.split('.n')[1] == 'py'):
        image = np.load(train_data_path_A + image_name)
        lenght_partial = image.shape[2]
    lenght = lenght + lenght_partial
print('image = ', image.shape, image.dtype, image.size, np.amax(image))
print('\n')
print(lenght)

IMG_HEIGHT = image.shape[0]
IMG_WIDTH =  image.shape[1]


''' X_train_A '''
X_train_A = np.zeros(((lenght), IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
count_images = 0
#images = []
for i, image_name in enumerate(sorted(train_list_A)):
    image_name = image_name.strip("._")      #This is for ERISXdl
    #print('\n image_name = ', image_name)
    if (image_name.split('.n')[1] == 'py'):
        image = np.load(train_data_path_A + image_name)
        #images = np.array(image)
        print(image_name, image.shape, image.dtype, image.size, np.amax(image))
        #print(image_name, images.shape, images.dtype, images.size, np.amax(images))
        for n in range (0 , image.shape[2]) :
            X_train_A[count_images] = image[:,:,n]
            count_images += 1

if IMG_CHANNELS == 3:
    X_train_A = np.stack([X_train_A, X_train_A, X_train_A], axis=3)   #Creating a 3channel matrix with a value of 3
    print('IMG_CHANNELS == 3')
if IMG_CHANNELS == 1:
    X_train_A = np.expand_dims(X_train_A, 3)    #Creating a 3channel matrix with a value of 1
    print('IMG_CHANNELS == 1')

print('X_train_A = ', X_train_A.shape, X_train_A.dtype, X_train_A.size, np.amax(X_train_A))
print('\n')


''' X_train_B '''
X_train_B = np.zeros(((lenght), IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
count_images = 0
#images = []
for i, image_name in enumerate(sorted(train_list_B)):
    image_name = image_name.strip("._")      #This is for ERISXdl
    #print('\n image_name = ', image_name)
    if (image_name.split('.n')[1] == 'py'):
        image = np.load(train_data_path_B + image_name)
        #images = np.array(image)
        print(image_name, image.shape, image.dtype, image.size, np.amax(image))
        #print(image_name, images.shape, images.dtype, images.size, np.amax(images))
        for n in range (0 , image.shape[2]) :
            X_train_B[count_images] = image[:,:,n]
            count_images += 1

if IMG_CHANNELS == 3:
    X_train_B = np.stack([X_train_B, X_train_B, X_train_B], axis=3)   #Creating a 3channel matrix with a value of 3
    print('IMG_CHANNELS == 3')
if IMG_CHANNELS == 1:
    X_train_B = np.expand_dims(X_train_B, 3)    #Creating a 3channel matrix with a value of 1
    print('IMG_CHANNELS == 1')

print('X_train_B = ', X_train_B.shape, X_train_B.dtype, X_train_B.size, np.amax(X_train_B))
print('\n')




train_A = tf.cast(X_train_A, dtype=tf.float32)
train_B = tf.cast(X_train_B, dtype=tf.float32)
print(train_A.shape, train_A.ndim, train_A.dtype)
print(train_B.shape, train_B.ndim, train_B.dtype)

sample_A = train_A[:1, :, :, :]
sample_B = train_B[:1, :, :, :]
print('Samples A and B = ', sample_A.shape, sample_A.ndim, sample_A.dtype, sample_B.shape, sample_B.ndim, sample_B.dtype)




######################  Figures  ##########################################
#BUFFER_SIZE = 1000
#BATCH_SIZE = 1
# normalizing the images to [-1, 1]

fig = plt.figure()
fig.add_subplot(121)
plt.title('TRAIN_A')
plt.imshow(sample_A[0])
fig.add_subplot(122)
plt.title('TRAIN_B')
plt.imshow(sample_B[0])
fig.savefig('1_TRAIN_sample.png')
#######################################################################################



######################  GENERATOR and DISCRIMINATOR  ##################################
OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

#######################################################################################


print('.....1.....')
print(sample_A.shape, sample_A.ndim, sample_A.dtype, sample_B.shape, sample_B.ndim, sample_B.dtype)
to_B = generator_g(sample_A)
print('.....2.....')
to_A = generator_f(sample_B)
print('.....3.....')
fig_generator = plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_A, to_B, sample_B, to_A]
title = ['A', 'To B', 'B', 'To A']

for i in range(len(imgs)):
  plt.subplot(2, 2, i+1)
  plt.title(title[i])
  if i % 2 == 0:
    #plt.imshow(imgs[i][0] * 0.5 + 0.5)
    plt.imshow(imgs[i][0])
  else:
    plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
#plt.show()
fig_generator.savefig('2_Generator.png')

fig_discriminator = plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real B?')
plt.imshow(discriminator_y(sample_B)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real A?')
plt.imshow(discriminator_x(sample_A)[0, ..., -1], cmap='RdBu_r')

#plt.show()
fig_discriminator.savefig('3_Discriminator.png')



#######################  LOSS FUNCTION  ##############################################
LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

#### Checkpoint  ###
checkpoint_path = "Checkpoint"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')
#######################################################################################

#######################  TRAINING  ####################################################

def generate_images(model, test_input):
    prediction = model(test_input)

    fig_true_vs_predicted = plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i])
        plt.axis('off')
    #plt.show()
    fig_true_vs_predicted.savefig('4_True_vs_Predicted.png')


@tf.function
def train_step(real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss,
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                              discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))

    return total_cycle_loss  ## AP



result = 0
for epoch in tqdm(range(EPOCHS)):
  start = time.time()

  n = 0
  for image_x_minonedim, image_y_minonedim in (zip(train_A, train_B)):
    image_x = tf.expand_dims(image_x_minonedim, axis=0)    #this expand the dimension from 3 to 4
    image_y = tf.expand_dims(image_y_minonedim, axis=0)    #this expand the dimension from 3 to 4
    #print('X', image_x.shape, image_x.ndim, image_x.dtype, 'Y', image_y.shape, image_y.ndim, image_y.dtype)
    result_v2 = train_step(image_x, image_y)   ## AP
    if n % 10 == 0:
      print ('.', end='')
    n += 1

  clear_output(wait=True)
  generate_images(generator_g, sample_A)   #generate an image of the generator and save it, just to see progresses

  if (epoch+1) == 1 :
    result = result_v2

  if (epoch+1) > 1 :
    if result_v2 < result:
      ckpt_save_path = ckpt_manager.save()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
  #if (epoch + 1) % 5 == 0:
   # ckpt_save_path = ckpt_manager.save()
    #print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
    #                                                     ckpt_save_path))

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))
  print ('\nLoss = ', result_v2)


start3 = datetime.now()
stop3  = datetime.now()
execution_time = stop3-start3
print("Execution time : ", execution_time)
print('Model saved')


######################################################################################################################
