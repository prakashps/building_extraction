# In[1]:

import numpy as np
from itertools import product
import rasterio as rio
import cv2
import os
from libtiff import TIFF
from libtiff import TIFFfile, TIFFimage

# In[3]:
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
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
#from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

# In[4]:

def iou(y_true, y_pred, smooth = 100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    #sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    iou_acc = (intersection + smooth) / (union + smooth)
    return iou_acc

  

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
  
# In[]:
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

# In[7]:
# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
  
# In[]:
#Num_training_samples = 2700
Num_Epochs = 50
Image_size = 96
tile_sizex = Image_size
tile_sizey = Image_size
#IMG_FILL_VALUE = 255
 
imgname_org = '. '
#imgname_orgmask = '. '
imgname_orgmask = None
img_tiles_path = './tiles/'
mask_tiles_path = './tiles/'
#mask_tiles_path = None
pred_tiles_path = './tiles/'
output_file_path = '. '
#morphed_output_file_path = '. '
thresh_output_file_path = '. '
#crf_output_file_path = '. '

model_path = './model.h5'

save_tiles = False
view_tiles = False
save_pred_tile = False
view_prediction = False
use_crf = False
use_threshold = True
mask_zero_thresh = 200
apply_morph = False
#NO_DATA_VAL = -3.4e+38
#NO_DATA_VAL = 0
NO_DATA_VAL = None
DEFAULT_VAL_BCKGRND = 0
DEFAULT_VAL_FORGRND = 255
#DEFAULT_VAL_BCKGRND = -108.66666
#DEFAULT_VAL_FORGRND = 4582.908

# In[]:
# Creating segments for input images
print('Reading Input image:',imgname_org)
tifinimg = TIFF.open(imgname_org)
inimg = tifinimg.read_image()
#inimg = cv2.imread(imgname_org)
print('Finished Reading Input image')
#inimg = cv2.cvtColor(inimg, cv2.COLOR_BGR2RGB)
img_dtype = type(inimg[0,0,0])
(H,W,Ch) = inimg.shape
#IMG_FILL_VALUE = np.min(inimg)
IMG_FILL_VALUE = np.max(inimg)

(H_delta,W_delta) = 0,0
if (H % tile_sizey) != 0:
  H_delta = tile_sizey - (H % tile_sizey)
if (W % tile_sizex) != 0:
  W_delta = tile_sizex - (W % tile_sizex)

top, bottom, left, right = 0, H_delta, 0, W_delta
inimg_new =  np.ones((H+H_delta, W+W_delta,Ch), dtype=img_dtype) * IMG_FILL_VALUE
inimg_new[:H,:W,:] = inimg

instd, inmean, inimg_new = mean_normalize(inimg_new)

offsets = product(range(0, W, tile_sizex), range(0, H, tile_sizey))
imageseg = dict()
trainx_list = []
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
  trainx_list.append(imgtile)
  
  cnt = cnt + 1
  
  if save_tiles == True:
    out_img_tile = os.path.join(img_tiles_path, 'img_{}-{}.tif'.format(row_off,col_off))
#    out_img_tile = './Austin/tiles_img_new/img_{}-{}.tif'.format(row_off,col_off)
    tif_img_tile = TIFF.open(out_img_tile, mode='w')
    tif_img_tile.write_image(reverse_mean_normalize(imgtile, instd, inmean).astype(img_dtype), compression='lzw', write_rgb=True)
    TIFF.close(tif_img_tile)
    
  if view_tiles_img == True:
    cv2.imshow("img_{}-{}".format(row_off,col_off), cv2.cvtColor(reverse_mean_normalize(imgtile, instd, inmean).astype(img_dtype),cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == 27:
      view_tiles_img = False
    
# Array of all the cropped Training sat Images    
trainx = np.asarray(trainx_list)

# In[]:
# Creating segments for input masks

if imgname_orgmask == None:
  (H2,W2) = (H,W)
  (H_delta2,W_delta2) = (H_delta,W_delta)
  mask_dtype = img_dtype
#  mask_dtype = np.uint8
  mask_minval = DEFAULT_VAL_BCKGRND
  mask_maxval = DEFAULT_VAL_FORGRND
#  mask_maxval = 255
else:
  tifinmask = TIFF.open(imgname_orgmask)
  inmask = tifinmask.read_image()
  #inmask = cv2.imread(imgname_orgmask)
  mask_dtype = type(inmask[0,0])
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
  trainy_list = []
  cnt2 = 1
  view_tiles_mask = view_tiles
  
  if (save_tiles == True) and (mask_tiles_path != None):
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
    
    imageseg2['{}-{}'.format(row_off2,col_off2)] = imgtile2
    trainy_list.append(imgtile2)
    
    cnt2 = cnt2 + 1
    
    if (save_tiles == True) and (mask_tiles_path != None):
      out_mask_tile = os.path.join(mask_tiles_path, 'mask_{}-{}.tif'.format(row_off2,col_off2))
  #    out_mask_tile = './Austin/tiles_mask_new/mask_{}-{}.tif'.format(row_off2,col_off2)
      tif_mask_tile = TIFF.open(out_mask_tile, mode='w')
      tif_mask_tile.write_image((imgtile2 * (mask_maxval - mask_minval)).astype(mask_dtype), compression='lzw')
      TIFF.close(tif_mask_tile)
    
    if view_tiles_mask == True:
      cv2.imshow("mask_{}-{}".format(row_off2,col_off2),(imgtile2 * (mask_maxval - mask_minval)).astype(mask_dtype))
      key = cv2.waitKey(0)
      cv2.destroyAllWindows()
      if key == 27:
        view_tiles_mask = False
      
  # Array of all the cropped Training gt Images    
  trainy = np.asarray(trainy_list)

# In[16]:

num_output_classes = 1

# In[24]:

inputs = Input(trainx.shape[1:])
n_classes=1
im_sz=Image_size
n_channels=Ch
n_filters_start=32
growth_factor=2
upconv=True
class_weights=[1.0]
  
droprate=0.25
n_filters = n_filters_start

conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

n_filters *= growth_factor
pool1 = BatchNormalization()(pool1)
conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
pool2 = Dropout(droprate)(pool2)

n_filters *= growth_factor
pool2 = BatchNormalization()(pool2)
conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
pool3 = Dropout(droprate)(pool3)

n_filters *= growth_factor
pool3 = BatchNormalization()(pool3)
conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool3)
conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_0)
pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4_0)
pool4_1 = Dropout(droprate)(pool4_1)

n_filters *= growth_factor
pool4_1 = BatchNormalization()(pool4_1)
conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_1)
conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_1)
pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_1)
pool4_2 = Dropout(droprate)(pool4_2)

n_filters *= growth_factor
conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_2)
conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv5)

n_filters //= growth_factor
if upconv:
    up6_1 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4_1])
else:
    up6_1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4_1])
up6_1 = BatchNormalization()(up6_1)
conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_1)
conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_1)
conv6_1 = Dropout(droprate)(conv6_1)

n_filters //= growth_factor
if upconv:
    up6_2 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_1), conv4_0])
else:
    up6_2 = concatenate([UpSampling2D(size=(2, 2))(conv6_1), conv4_0])
up6_2 = BatchNormalization()(up6_2)
conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_2)
conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_2)
conv6_2 = Dropout(droprate)(conv6_2)

n_filters //= growth_factor
if upconv:
    up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_2), conv3])
else:
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv3])
up7 = BatchNormalization()(up7)
conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv7)
conv7 = Dropout(droprate)(conv7)

n_filters //= growth_factor
if upconv:
    up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
else:
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
up8 = BatchNormalization()(up8)
conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv8)
conv8 = Dropout(droprate)(conv8)

n_filters //= growth_factor
if upconv:
    up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
else:
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv9)

conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)

model = Model(inputs=inputs, outputs=conv10)

model.summary()

# In[25]:

#model.compile(optimizer = Adam(lr = 0.00001), loss = 'binary_crossentropy', metrics = ['accuracy', iou])
#model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
#model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=['accuracy', iou, dice_coef])
#model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
model.compile(optimizer=Adam(), loss=weighted_binary_crossentropy, metrics=['accuracy', iou, dice_coef, jaccard_coef])

# In[ ]:

print('Loading weights')
model.load_weights(model_path)

# In[ ]:

final_imgmask = np.zeros((H+H_delta,W+W_delta),dtype=img_dtype)

# In[ ]:
# List of file names of actual Satellite images for testing 
filelist_testx = imageseg.keys()
Num_test_samples = len(filelist_testx)

if os.path.exists(pred_tiles_path):
  print('Output directory exists. Deleting contents..')
  for old_file in os.listdir(pred_tiles_path):
    old_file_path = os.path.join(pred_tiles_path, old_file)
    os.remove(old_file_path)
else:
  print('Creating output directory:',pred_tiles_path)
  os.mkdir(pred_tiles_path)

for idx, fname in enumerate(filelist_testx):
    test_image_name = fname
    
    print('Processing tile ({}/{}): (roff-coff)=({})'.format(idx+1, Num_test_samples, test_image_name))
    # Reading the image
    im = imageseg[fname]

#    im = np.expand_dims(im, axis=0) # shape (1, x_pixels, y_pixels, n_bands)
    im = im[np.newaxis,:] # shape (1, x_pixels, y_pixels, n_bands)
    
    pred_mask_arr = model.predict(im)
    
    pred_mask = np.reshape(pred_mask_arr[0],(pred_mask_arr[0].shape[0],pred_mask_arr[0].shape[1]))
    
    test_mask_name = test_image_name
    
#    if imgname_orgmask != None:
#      # Getting the actual mask
#      actual_mask = imageseg2[test_mask_name]
#      cv2.imshow("actual mask",actual_mask)
      
    if save_pred_tile == True:
      out_file_name = pred_tiles_path + str(test_image_name) + '_pred.tif'

      tif_pred = TIFF.open(out_file_name, mode='w')
      tif_pred.write_image(pred_mask)
      TIFF.close(tif_pred)
    
    imgname_parts = test_mask_name.split('-')
    tile_w, tile_h  = int(imgname_parts[0]), int(imgname_parts[1])

    final_imgmask[tile_h:tile_h+pred_mask.shape[0],tile_w:tile_w+pred_mask.shape[1]] = (pred_mask*(mask_maxval - mask_minval)).astype(mask_dtype)
    
#    cv2.imshow("image", cv2.cvtColor(reverse_mean_normalize(im, instd, inmean),cv2.COLOR_RGB2BGR))
#    cv2.imshow("pred_mask",(pred_mask*(mask_maxval - mask_minval)).astype(mask_dtype))

#    cv2.imshow("final_imgmask(600x600)",final_imgmask[:600,:600])
#    cv2.imshow("actual_imgmask(600x600)",inmask[:600,:600])
#    
#    key = cv2.waitKey(0)
#    if key == 27:
#      cv2.destroyAllWindows()
#      break
#    cv2.destroyAllWindows()
    
# In[ ]:
## Create prediction file without metadata
#out_file_name_final = './Austin/austin_gt_pred.tif'
#if os.path.exists(out_file_name_final):
#  print('Removing old file:',out_file_name_final)
#  os.remove(out_file_name_final)
#tif_pred_final = TIFF.open(out_file_name_final, mode='w')
#tif_pred_final.write_image(final_imgmask, compression='lzw')
#TIFF.close(tif_pred_final)

# In[ ]:
# Create prediction file with metadata
out_file_name_final2 = output_file_path
if os.path.exists(out_file_name_final2):
  print('Removing old file:',out_file_name_final2)
  os.remove(out_file_name_final2)
  
if imgname_orgmask != None:
  print('Using mask image to read metadata:',imgname_orgmask)
  inds = rio.open(imgname_orgmask)
else:
  print('Using input image to read metadata:',imgname_org)
  inds = rio.open(imgname_org)
  
meta = inds.meta.copy()
meta['compress']='lzw'
meta['count'] = 1
meta['nodata'] = NO_DATA_VAL
inds.close()
outds = rio.open(out_file_name_final2, 'w', **meta)
outds.write(final_imgmask[:H2,:W2],meta['count'])
outds.close()

print('Finished processing')

if view_prediction == True:
  cv2.imshow("final_imgmask(600x600)",final_imgmask[:H2,:W2][-600:,-600:])
  if imgname_orgmask != None:
    cv2.imshow("actual_imgmask(600x600)",inmask[-600:,-600:])
  
  print("Generated output image:", out_file_name_final2)
  key = cv2.waitKey(0)
  cv2.destroyAllWindows()

print('Finished processing')

# In[ ]:

if use_threshold == True:
  print('Getting Threshold output')
  
  if os.path.exists(thresh_output_file_path):
    print('Removing old file:',thresh_output_file_path)
    os.remove(thresh_output_file_path)
  #  
  thresh_mask = final_imgmask[:H2,:W2].copy()
  #print('%white pixels before thresh:', 100 * (np.sum(thresh_mask > 0))/(H2 * W2))
  thresh_mask[final_imgmask[:H2,:W2] <= (mask_maxval * mask_zero_thresh)] = mask_minval
  thresh_mask[final_imgmask[:H2,:W2] > (mask_maxval * mask_zero_thresh)] = mask_maxval
  #print('%white pixels after thresh:', 100 * (np.sum(thresh_mask > 0))/(H2 * W2))
  #  
  outds3 = rio.open(thresh_output_file_path, 'w', **meta)
  outds3.write(thresh_mask, meta['count'])
  outds3.close()
  print('Threshold output file generated:', thresh_output_file_path)


if use_crf == True:
  # Add CRF
  
  import pydensecrf.densecrf as dcrf
  from pydensecrf.utils import unary_from_labels
  #from pydensecrf.utils import create_pairwise_bilateral
  from pydensecrf.utils import unary_from_softmax
  
  # CRF params
  
  # Experimenting with CRF post processing function
  def crf_try(im, mask, zero_unsure=True):
      colors, labels = np.unique(mask, return_inverse=True)
      probs = np.tile(mask[np.newaxis,:,:],(2,1,1))
      probs[1,:,:] = 1 - probs[0,:,:]
      # Inference without pair-wise terms
      U = unary_from_softmax(probs)  # note: num classes is first dim
      n_labels = 2
      #Setting up the CRF model
      d = dcrf.DenseCRF2D(im.shape[1], im.shape[0], n_labels)
      d.setUnaryEnergy(U)
      # This adds the color-independent term, features are the locations only.
      d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
      
      # Create simple image which will serve as bilateral.
      # Note that we put the channel dimension last here,
      # but we could also have it be the first dimension and
      # just change the `chdim` parameter to `0` further down.
  #    (H, W) = mask.shape[:2]
  #    NCHAN=1
  #    img = np.zeros((H,W,NCHAN), np.uint8)
  #    img[H//3:2*H//3,W//4:3*W//4,:] = 1
      
      # Create the pairwise bilateral term from the above image.
      # The two `s{dims,chan}` parameters are model hyper-parameters defining the strength of the location and image content bilaterals, respectively.
      # pairwise_energy now contains as many dimensions as the DenseCRF has features, which in this case is 3: (x,y,channel1)
  #    pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=(0.01,), img=img, chdim=2)
  #    d.addPairwiseEnergy(pairwise_energy, compat=10)  # `compat` is the "strength" of this potential.
      #Run Inference
      Q = d.inference(10)
      # Find out the most probable class for each pixel.
      MAP = np.argmax(Q, axis=0)
      MAP = 1 - MAP
      return MAP.reshape((im.shape[0],im.shape[1]))
  
  # Fully connected CRF post processing function
  def do_crf(im, mask, zero_unsure=True):
      colors, labels = np.unique(mask, return_inverse=True)
      image_size = mask.shape[:2]
      n_labels = len(set(labels.flat))
      d = dcrf.DenseCRF2D(image_size[1], image_size[0], n_labels)  # width, height, nlabels
      U = unary_from_labels(labels, n_labels, gt_prob=.7, zero_unsure=zero_unsure)
      d.setUnaryEnergy(U)
      # This adds the color-independent term, features are the locations only.
      d.addPairwiseGaussian(sxy=(3,3), compat=3)
      # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
      # im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
      
      d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=im.astype('uint8'), compat=10)
      Q = d.inference(5) # 5 - num of iterations
      
      MAP = np.argmax(Q, axis=0).reshape(image_size)
      unique_map = np.unique(MAP)
      for u in unique_map: # get original labels back
          np.putmask(MAP, MAP == u, colors[u])
      return MAP
      # MAP = do_crf(frame, labels.astype('int32'), zero_unsure=False)
  
  """
  Function which returns the labelled image after applying CRF
  
  """
  #Original_image = Image which has to labelled
  #Mask image = Which has been labelled by some technique..
  def crf(original_image, mask_img):
      
      # Converting annotated image to RGB if it is Gray scale
  #    if(len(mask_img.shape)<3):
  #        mask_img = gray2rgb(mask_img)
  
  #     #Converting the annotations RGB color to single 32 bit integer
  #    annotated_label = mask_img[:,:,0] + (mask_img[:,:,1]<<8) + (mask_img[:,:,2]<<16)
      annotated_label = mask_img
      
  #     # Convert the 32bit integer color to 0,1, 2, ... labels.
      colors, labels = np.unique(annotated_label, return_inverse=True)
  
      n_labels = 2
      
      #Setting up the CRF model
      d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)
  
      # get unary potentials (neg log probability)
      U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
      d.setUnaryEnergy(U)
  
      # This adds the color-independent term, features are the locations only.
      d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
          
      #Run Inference for 10 steps 
      Q = d.inference(10)
  
      # Find out the most probable class for each pixel.
      MAP = np.argmax(Q, axis=0)
  
      return MAP.reshape((original_image.shape[0],original_image.shape[1]))
  


  print('Getting CRF output')
  final_imgmask_short = final_imgmask[:H2,:W2].copy()
  #crf_output = np.float32(do_crf(inimg, thresh_mask, zero_unsure=False))
  #crf_output = np.float32(crf(inimg, thresh_mask))
  crf_output = np.float32(crf_try(inimg, final_imgmask_short))
  
  print('Finished getting CRF output')
  
  #cv2.imshow("crf_output(600x600)",crf_output[-600:,-600:])
  
  if os.path.exists(crf_output_file_path):
    print('Removing old file:',crf_output_file_path)
    os.remove(crf_output_file_path)
    
  outds2 = rio.open(crf_output_file_path, 'w', **meta)
  outds2.write(crf_output, meta['count'])
  outds2.close()
  print('CRF output file generated:',crf_output_file_path)


# In[ ]:

if apply_morph == True:
  # Apply morphology operations
  dilateSize = 3
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilateSize,dilateSize))
  #thresh_mask_new = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, kernel)
  #thresh_mask_new = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel)
  thresh_mask_new = cv2.erode(thresh_mask,kernel,iterations=2)
  #thresh_mask_new = cv2.dilate(thresh_mask,kernel,iterations=2)
  
  
  if os.path.exists(morphed_output_file_path):
    print('Removing old file:',morphed_output_file_path)
    os.remove(morphed_output_file_path)
  
  outds4 = rio.open(morphed_output_file_path, 'w', **meta)
  outds4.write(thresh_mask_new, meta['count'])
  outds4.close()
  print('Morphed output file generated:', morphed_output_file_path)

# In[ ]:
# Confusion Matrix calculation
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
  
actual = None
predicted = None
  
if imgname_orgmask != None:
  actual = inmask.copy()
else:
  print('Actual mask not available')

if use_threshold == True:
  predicted = thresh_mask.copy()
elif use_crf == True:
  predicted = crf_output.copy()
else:
  print('Prediction output not available')

if type(actual) != type(None) and type(predicted) != type(None):
  print('Calculating confusion matrix')
  results = confusion_matrix(actual.flatten(), predicted.flatten())
  print ('Confusion Matrix :')
  print(results)
  print ('Accuracy Score :', accuracy_score(actual.flatten(), predicted.flatten()))
  print ('Report : ')
  print (classification_report(actual.flatten(), predicted.flatten()))
else:
  print('Cannot calculate confusion matrix')
