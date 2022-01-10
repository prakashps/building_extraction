# In[]:
#import time


# In[1]:

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import sys
from itertools import product
import cv2
import os
import matplotlib.pyplot as plt
from libtiff import TIFF
from keras.models import Model
from keras.layers import Conv2D #to add convolution layers
from keras.layers import MaxPooling2D # to add pooling layers
from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import Dropout 
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.initializers import glorot_uniform
from keras.initializers import he_uniform
from keras.initializers import he_normal
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

# In[]:
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
 

# In[4]:
start = time.process_time()

def iou(y_true, y_pred, smooth = 100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    #sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    iou_acc = (intersection + smooth) / (union + smooth)
    return iou_acc


# In[5]:

def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

# In[6]:

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# In[6]:

def weighted_binary_crossentropy(y_true, y_pred):
    class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
    return K.sum(class_loglosses * K.constant(class_weights))

# In[7]:
# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
  
# In[]:

def mean_normalize_dummy(image):
  std_img = np.std(image)
  mean_img = np.mean(image)
  norm_img = image
  return std_img, mean_img, norm_img

def mean_normalize_org(image):
  std_img = np.std(image)
  mean_img = np.mean(image)
  norm_img = (image - mean_img)/std_img
  return std_img, mean_img, norm_img

def mean_normalize(image):
  std_lst = []
  mean_lst = []
  for idx in range(image.shape[2]):
    std_chan = np.std(image[:,:,idx])
    mean_chan = np.mean(image[:,:,idx])
    std_lst.append(std_chan)
    mean_lst.append(mean_chan)
    
  std_img = np.array([std_lst])
  mean_img = np.array([mean_lst])
  norm_img = (image - mean_img)/std_img
  return std_img, mean_img, norm_img

def reverse_mean_normalize(norm, std_img, mean_img):
  image = (norm * std_img) + mean_img
  return image

def reverse_mean_normalize_dummy(norm, std_img, mean_img):
  return norm

# In[7]:

IMAGE_SIZE = 96

imgname_org = '. '
imgname_orgmask = '. '
img_tiles_path = '. '
mask_tiles_path = '. '

# define the path to the output learning rate finder plot, training
# history plot and cyclical learning rate plot
LRFIND_PLOT_PATH = os.path.sep.join(["logs", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["logs", "training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join(["logs", "clr_plot.png"])

# Set training list size as fraction of available data
Num_Epochs = 50
tile_sizex = IMAGE_SIZE
tile_sizey = IMAGE_SIZE

log_folder = './logs/'
checkpoints_dir = './checkpoints/'
model_path = './model.h5'
plotPath = os.path.sep.join([log_folder, "unet_train_plot.png"])
jsonPath = os.path.sep.join([log_folder, "unet_train_json.json"])

save_logs = True
show_graphs = True
save_tiles = False
view_tiles = False
augment_images = False
use_callbacks = False
load_model_weights = True # first step config
#load_model_weights = False # second step config
lr_find = False
use_clr = False

#IMG_FILL_VALUE = 255
#init_learning_rate = 1e-2 # trial with SGD
init_learning_rate = 1e-3 # first step learning rate
#init_learning_rate = 1e-5 # second step learning rate
min_feature_cnt = 1
train_drop_rate = 0.2
train_batch_size = 16
start_epoch = 0
save_weights_periodicity = 10
CLR_MIN_LR = 1e-5
CLR_MAX_LR = 1e-2
CLR_STEP_SIZE = 8
CLR_METHOD = "triangular2" #"triangular"/"triangular2"/"exp_range"
lr_schedule = None #None/"step"/"linear"/"poly"

# In[7]:

# Creating segments for input images
tifinimg = TIFF.open(imgname_org)
inimg = tifinimg.read_image()
TIFF.close(tifinimg)

img_dtype = type(inimg[0,0,0])
(H,W,Ch) = inimg.shape
(H_delta,W_delta) = 0,0
if (H % tile_sizey) != 0:
  H_delta = tile_sizey - (H % tile_sizey)
if (W % tile_sizex) != 0:
  W_delta = tile_sizex - (W % tile_sizex)

#IMG_FILL_VALUE = np.min(inimg)
IMG_FILL_VALUE = np.max(inimg)

top, bottom, left, right = 0, H_delta, 0, W_delta
inimg_new =  np.ones((H+H_delta, W+W_delta,Ch), dtype=img_dtype) * IMG_FILL_VALUE
inimg_new[:H,:W,:] = inimg
#inimg_new =  cv2.copyMakeBorder(inimg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=IMG_FILL_VALUE)

instd_img, inmean_img, inimg_new = mean_normalize(inimg_new)

offsets = product(range(0, W, tile_sizex), range(0, H, tile_sizey))
imageseg = dict()
#trainx_list = []
cnt = 1
view_tiles_img = view_tiles

if save_tiles == True:
  if os.path.exists(img_tiles_path):
    print('Image tiles directory exists. Deleting contents..')
    for old_img_tile in os.listdir(img_tiles_path):
      old_img_tile_p = os.path.join(img_tiles_path, old_img_tile)
      os.remove(old_img_tile_p)
  else:
    print('Creating Image tiles directory:',img_tiles_path)
    os.mkdir(img_tiles_path)
    
for row_off,col_off in offsets:
  print("Img:{}. (Row:Col) = ({}:{})".format(cnt,row_off,col_off))
  
  col_start, col_end, row_start, row_end = col_off, col_off+tile_sizey-1, row_off, row_off+tile_sizex-1
  
  imgtile = inimg_new[col_start:col_end+1,row_start:row_end+1,:]
  
  imageseg['{}-{}'.format(row_off,col_off)] = imgtile
#  trainx_list.append(imgtile)
  
  cnt = cnt + 1
  
  if save_tiles == True:
#    print('imgtile.shape:',imgtile.shape)
    out_img_tile = os.path.join(img_tiles_path, 'img_{}-{}.tif'.format(row_off,col_off))
#    out_img_tile = './Austin/tiles_img_new/img_{}-{}.tif'.format(row_off,col_off)
    tif_img_tile = TIFF.open(out_img_tile, mode='w')
    tif_img_tile.write_image(reverse_mean_normalize(imgtile, instd_img, inmean_img).astype(img_dtype), compression='lzw', write_rgb=True)
    TIFF.close(tif_img_tile)
    
  if view_tiles_img == True:
    cv2.imshow("img_{}-{}".format(row_off,col_off), cv2.cvtColor(reverse_mean_normalize(imgtile, instd_img, inmean_img).astype(img_dtype),cv2.COLOR_RGB2BGR))
    
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == 27:
      view_tiles_img = False

# In[]:
# Creating segments for input masks

tifinmask = TIFF.open(imgname_orgmask)
inmask = tifinmask.read_image()
#inmask = cv2.imread(imgname_orgmask)
TIFF.close(tifinmask)

mask_dtype = type(inmask[0,0])
# Assuming here that mask has only 1 channel
(H2,W2) = inmask.shape
(H_delta2,W_delta2) = 0,0
if (H2 % tile_sizey) != 0:
  H_delta2 = tile_sizey - (H2 % tile_sizey)
if (W2 % tile_sizex) != 0:
  W_delta2 = tile_sizex - (W2 % tile_sizex)

top2, bottom2, left2, right2 = 0, H_delta2, 0, W_delta2
inmask_new =  cv2.copyMakeBorder(inmask, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=0)

mask_minval = np.min(inmask_new)
mask_maxval = np.max(inmask_new)
inmask_new = inmask_new / (mask_maxval - mask_minval)
#cv2.imwrite(outpath, inimg_new)

offsets2 = product(range(0, W2, tile_sizex), range(0, H2, tile_sizey))
imageseg2 = dict()
trainx_list = []
trainy_list = []
cnt2 = 1
view_tiles_mask = view_tiles

if save_tiles == True:
  if os.path.exists(mask_tiles_path):
    print('Mask tiles directory exists. Deleting contents..')
    for old_mask_tile in os.listdir(mask_tiles_path):
      old_mask_tile_p = os.path.join(mask_tiles_path, old_mask_tile)
      os.remove(old_mask_tile_p)
  else:
    print('Creating Mask tiles directory:',mask_tiles_path)
    os.mkdir(mask_tiles_path)

for row_off2,col_off2 in offsets2:
  print("Mask:{}. (Row:Col) = ({}:{})".format(cnt2,row_off2,col_off2))
  
  col_start2, col_end2, row_start2, row_end2 = col_off2, col_off2+tile_sizey-1, row_off2, row_off2+tile_sizex-1
  
  imgtile2 = inmask_new[col_start2:col_end2+1,row_start2:row_end2+1]
  
  imgtile2_org = imgtile2.copy()
  imgtile2 = np.expand_dims(imgtile2, axis=2) # shape (1, x_pixels, y_pixels, n_bands)
  imageseg2['{}-{}'.format(row_off2,col_off2)] = imgtile2
  
  # Consider for training only if atleast some features are present in the tile
  if (cv2.countNonZero(imageseg2['{}-{}'.format(row_off2,col_off2)])) > min_feature_cnt:
    
    trainx_list.append(imageseg['{}-{}'.format(row_off2,col_off2)])
    
    trainy_list.append(imgtile2)
  
  cnt2 = cnt2 + 1
  
  if save_tiles == True:
#    print('imgtile2.shape:',imgtile2.shape)
    out_mask_tile = os.path.join(mask_tiles_path, 'mask_{}-{}.tif'.format(row_off2,col_off2))
#    out_mask_tile = './Austin/tiles_mask_new/mask_{}-{}.tif'.format(row_off2,col_off2)
    tif_mask_tile = TIFF.open(out_mask_tile, mode='w')
    tif_mask_tile.write_image((imgtile2_org * (mask_maxval - mask_minval)).astype(mask_dtype), compression='lzw')
    TIFF.close(tif_mask_tile)
  
  if view_tiles_mask == True:
    cv2.imshow("mask_{}-{}".format(row_off2,col_off2),(imgtile2_org * (mask_maxval - mask_minval)).astype(mask_dtype))
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == 27:
      view_tiles_mask = False
    
trainx = np.asarray(trainx_list)
trainy = np.asarray(trainy_list)

# In[16]:
#num_output_classes = dfy.shape[-1]
num_output_classes = 1
print(num_output_classes)

# In[17]:

print('trainx.shape:',trainx.shape)
print('trainy.shape:',trainy.shape)
min_samples = min(trainx.shape[0],trainy.shape[0])
#x_train, x_test, y_train, y_test = train_test_split(dx, dfy, test_size=0.2, random_state=4)
x_train, x_test, y_train, y_test = train_test_split(trainx[:min_samples], trainy[:min_samples], test_size=0.30, random_state=4)

# In[18]:
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(x_train.shape[1:])

# In[24]:

inputs = Input(x_train.shape[1:])
n_classes=1
im_sz=IMAGE_SIZE
n_channels=Ch
n_filters_start=32
growth_factor=2
upconv=True
class_weights=[1.0]
#droprate=0.25
droprate = train_drop_rate
BatchSize = train_batch_size

#kernel_init=glorot_uniform()
kernel_init=he_uniform()
#kernel_init=he_normal()

kernel_reg=None
#kernel_reg=l2(0.0005)

n_filters = n_filters_start
#inputs = Input((im_sz, im_sz, n_channels))
#inputs = BatchNormalization()(inputs)

#keras.initializers.he_uniform(seed=None)
# 32
conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(inputs)
conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#pool1 = Dropout(droprate)(pool1)

# 64
n_filters *= growth_factor
pool1 = BatchNormalization()(pool1)
conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(pool1)
conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
pool2 = Dropout(droprate)(pool2)

# 128
n_filters *= growth_factor
pool2 = BatchNormalization()(pool2)
conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(pool2)
conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
pool3 = Dropout(droprate)(pool3)

# 256
n_filters *= growth_factor
pool3 = BatchNormalization()(pool3)
conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(pool3)
conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(conv4_0)
pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4_0)
pool4_1 = Dropout(droprate)(pool4_1)

# 512
n_filters *= growth_factor
pool4_1 = BatchNormalization()(pool4_1)
conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(pool4_1)
conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(conv4_1)
pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_1)
pool4_2 = Dropout(droprate)(pool4_2)

# 1024
n_filters *= growth_factor
conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(pool4_2)
conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(conv5)

# 512
n_filters //= growth_factor
if upconv:
    up6_1 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(conv5), conv4_1])
else:
    up6_1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4_1])
up6_1 = BatchNormalization()(up6_1)
conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(up6_1)
conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(conv6_1)
conv6_1 = Dropout(droprate)(conv6_1)

# 256
n_filters //= growth_factor
if upconv:
    up6_2 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(conv6_1), conv4_0])
else:
    up6_2 = concatenate([UpSampling2D(size=(2, 2))(conv6_1), conv4_0])
up6_2 = BatchNormalization()(up6_2)
conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(up6_2)
conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(conv6_2)
conv6_2 = Dropout(droprate)(conv6_2)

# 128
n_filters //= growth_factor
if upconv:
    up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(conv6_2), conv3])
else:
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv3])
up7 = BatchNormalization()(up7)
conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(up7)
conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(conv7)
conv7 = Dropout(droprate)(conv7)

# 64
n_filters //= growth_factor
if upconv:
    up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(conv7), conv2])
else:
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
up8 = BatchNormalization()(up8)
conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(up8)
conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(conv8)
conv8 = Dropout(droprate)(conv8)

# 32
n_filters //= growth_factor
if upconv:
    up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(conv8), conv1])
else:
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(up9)
conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(conv9)

conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(conv9)

model = Model(inputs=inputs, outputs=conv10)

 

model.compile(
    optimizer=Adam(lr=init_learning_rate, decay=init_learning_rate / Num_Epochs), 
    loss=weighted_binary_crossentropy, 
    metrics=['accuracy', iou, dice_coef, jaccard_coef])

 

model.summary()

# In[ ]:

if load_model_weights == True and os.path.exists(model_path):
  print('Loading weights....')
  model.load_weights(model_path)

# In[ ]:

callbacks = None
callback_option = 1

if use_callbacks == True:
  if callback_option == 1:
      pass
#    callbacks = [
#      EpochCheckpoint(checkpoints_dir, 
#                      every=save_weights_periodicity,
#                      startAt=start_epoch),
#      TrainingMonitor(plotPath,
#                      jsonPath=jsonPath,
#                      startAt=start_epoch)]
  elif callback_option == 2:
    callbacks = [
            ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='min'),
            #EarlyStopping(monitor='val_loss', patience=20, verbose=1),
            #ModelCheckpoint('zf_unet_224_temp.h5', monitor='val_loss', save_best_only=True, verbose=0),
            ]
  else:
      pass
# In[ ]:
    
# check to see if we are attempting to find an optimal learning rate
# before training for the full number of epochs
if lr_find == True:
  # initialize the learning rate finder and then train with learning
  # rates ranging from 1e-10 to 1e+1
  print("[INFO] finding learning rate...")
#  lrf = LearningRateFinder(model)
#  lrf.find(
#    [x_train, y_train],
#    1e-10, 1e+1,
#    stepsPerEpoch=None,
#    batchSize=BatchSize,
#    epochs=10)
  
  # plot the loss for the various learning rates and save the
  # resulting plot to disk
#  lrf.plot_loss()
  plt.savefig(LRFIND_PLOT_PATH)
  
  # gracefully exit the script so we can adjust our learning rates
  # in the config and then train the network for our full set of
  # epochs
  print("[INFO] learning rate finder complete")
  print("[INFO] examine plot and adjust learning rates before training")
  sys.exit(0)

# In[ ]:
if use_clr == True:
  # otherwise, we have already defined a learning rate space to train
  # over, so compute the step size and initialize the cyclic learning
  # rate method
  print("[INFO] Using Cyclic learning rates...")
  stepSize = CLR_STEP_SIZE * (x_train.shape[0] // BatchSize)
#  clr = CyclicLR(
#    mode=CLR_METHOD,
#    base_lr=CLR_MIN_LR,
#    max_lr=CLR_MAX_LR,
#    step_size=stepSize)
#  
#  if use_callbacks == True:
#    callbacks.append(clr)
#  else:
#    callbacks = [clr]

# In[ ]:
# check to see if step-based learning rate decay should be used
schedule = None
#if lr_schedule == "step":
#  print("[INFO] using 'step-based' learning rate decay...")
#  schedule = StepDecay(initAlpha=init_learning_rate, factor=0.25, dropEvery=15)
## check to see if linear learning rate decay should should be used
#elif lr_schedule == "linear":
#  print("[INFO] using 'linear' learning rate decay...")
#  schedule = PolynomialDecay(maxEpochs=Num_Epochs, initAlpha=init_learning_rate, power=1)
## check to see if a polynomial learning rate decay should be used
#elif lr_schedule == "poly":
#  print("[INFO] using 'polynomial' learning rate decay...")
#  schedule = PolynomialDecay(maxEpochs=Num_Epochs, initAlpha=init_learning_rate, power=5)

# if the learning rate schedule is not empty, add it to the list of
# callbacks
#if schedule is not None:
#  if use_callbacks == True:
#    callbacks.append(LearningRateScheduler(schedule))
#  else:
#    callbacks = [LearningRateScheduler(schedule)]

# In[ ]:

if augment_images == True:
  print("Performing 'on the fly' data augmentation")
  aug = ImageDataGenerator(
          #rotation_range=10,
          #zoom_range=0.50,
          width_shift_range=0.1,
          height_shift_range=0.1,
          #shear_range=0.15,
          horizontal_flip=True,
          vertical_flip=True,
          fill_mode="nearest")
  
  print("Training network for {} epochs...".format(Num_Epochs))
  history = model.fit_generator(
        aug.flow(x_train, y_train, batch_size=BatchSize),
        validation_data=(x_test, y_test),
        steps_per_epoch=len(x_train) // BatchSize,
        validation_steps=len(x_train) // BatchSize,
        epochs=Num_Epochs,
        verbose=1,
        shuffle=True,
        callbacks=callbacks)
else:
  print("Training network for {} epochs...".format(Num_Epochs))
  #history = model.fit(x_train, y_train, epochs=Num_Epochs, steps_per_epoch=1, validation_split=0.2, validation_steps=1, batch_size=None, verbose=1, callbacks=callbacks)
  #history = model.fit(x_train, y_train, epochs=Num_Epochs, validation_split=0.2, batch_size=32, verbose=1, callbacks=callbacks)
  #history = model.fit(x_train, y_train, epochs=Num_Epochs, validation_data = (x_test, y_test), batch_size=BatchSize, verbose=1,shuffle=True,steps_per_epoch=None, callbacks=callbacks)
  history = model.fit(x_train, y_train, epochs=Num_Epochs, validation_data = (x_test, y_test), batch_size=BatchSize, verbose=1,shuffle=True,steps_per_epoch=None,callbacks=callbacks)

# In[ ]:

if save_logs == True:
  curtime = time.asctime(time.localtime())
  mnval = curtime.split()[1]
  dtval = curtime.split()[2]
  tmval = re.sub(':','-',curtime.split()[3])
  yrval = curtime.split()[4]
  if not os.path.exists(log_folder):
    os.mkdir(log_folder)
  log_file_name = os.path.join(log_folder, 'unet_train_log_{}{}{}_{}.csv'.format(dtval,mnval,yrval,tmval))
  
  print("Saving training logs in file:", log_file_name)
  pd.DataFrame(history.history).to_csv(log_file_name, index=False)

# In[ ]:
  
if show_graphs == True:
  # list all data in history
#  print(history.history.keys())
  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(os.path.join(log_folder,'acc_plot.png'))
  plt.show()
#  plt.close()
  # summarize history for loss
  plt.figure()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['train', 'val'], loc='upper right')
  plt.savefig(os.path.join(log_folder,'loss_plot.png'))
  plt.show()
#  plt.close()
  
  # summarize history for dice coeff
  plt.figure()
  plt.plot(history.history['dice_coef'])
  plt.plot(history.history['val_dice_coef'])
  plt.title('Model dice_coeff')
  plt.ylabel('dice_coef')
  plt.xlabel('Epoch')
  plt.legend(['train', 'val'], loc='upper right')
  plt.savefig(os.path.join(log_folder,'dice_coef_plot.png'))
  plt.show()
#  plt.close()

# In[ ]:
if os.path.exists(model_path):
  print("Deleting existing model file")
  os.remove(model_path)

print("Saving model file:",model_path)
model.save(model_path, overwrite=True)

# In[ ]:
print('Finished model training')

print(time.process_time() - start)