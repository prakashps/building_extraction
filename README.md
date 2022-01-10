# Building extraction - A deep learning approach
A deep learning pipeline for extraction building footprints from high resolution remote sensing imagery

Dependencies

  sklearn
  pandas
  numpy
  matplotlib
  rasterio
  itertools
  cv2
  os
  time
  sys
  libtiff
  keras
  tensorflow


Steps:

- Use "Model_training.py" to train the model, that creates image tiles and trained model in .h5 (model.h5) format
- Use "Model_prediction" for prediction from the trained model model.h5 
- Both for training and testing images and masks should be rectangular without any zero or null or no data values
- Please use "image.tif" and "mask.tif" should be of same dimension. (Ex: Image: 5000X5000X3 & Mask: 5000X5000X1) 
