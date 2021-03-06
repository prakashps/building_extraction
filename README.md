# Building extraction - A deep learning approach
A complete deep learning pipeline for deriving building footprints from high-resolution remote sensing imagery.

## Citation

Prakash P.S., & Aithal, B. H. (2022). Building footprint extraction from very high-resolution satellite images using deep learning. Journal of Spatial Science, 1-17. https://doi.org/10.1080/14498596.2022.2037473


## Dependencies

  sklearn,
  pandas,
  numpy,
  matplotlib,
  rasterio,
  itertools,
  cv2,
  os,
  time,
  sys,
  libtiff,
  keras,
  tensorflow



## Steps:

- Use "Model_training.py" to train the model, that creates image tiles and trained model in .h5 (model.h5) format
- Use "Model_prediction.py" for prediction from the trained model model.h5 
- Both for training and testing images and masks should be rectangular without any zero or null or no data values
- Please use "image.tif" and "mask.tif" should be of same dimension. (Ex: Image: 5000X5000X3 & Mask: 5000X5000X1) 

