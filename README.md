# CNST-6308-Roadside Vegetation Detection using Dashcam

This project aims to detect roadside vegetation from dashcam images using the MIT DriveSeg dataset and the pre-trained MobileNetV2_Coco_Cityscapes_trainfine model. The project utilizes the semantic segmentation technique to identify the vegetation in the input image. The MobileNetV2_Coco_Cityscapes_trainfine model is a state-of-the-art deep learning model trained on the Cityscapes dataset that includes a diverse range of urban street scenes, including vegetation.

**DeepLabv3+** is a state-of-the-art deep learning model for semantic segmentation tasks, such as image classification and object detection. It was introduced by Google Research in 2018 and builds upon the previous versions of DeepLab by incorporating an encoder-decoder architecture, atrous spatial pyramid pooling, and deep supervision.

![image](https://github.com/kundamnikhil/CNST-MIT-Drive-Segmentation/assets/43941418/ac9915e4-f6e8-4f63-ab33-1fea5ac9b571)

## Implementation

### Step 1: Install Required Libraries and Load the Libraries
- TensorFlow
- OS
- BytesIO
- Tarfile
- Tempfile
- Urllib
- Six.moves
- Matplotlib
- Gridspec
- Pyplot
- Numpy
- PIL
- Cv2
- Tqdm
- IPython
- Sklearn.metrics
- Tabulate
- Warnings

### Step 2: Create a Model Segmentation
- Define the DeepLabModel class with init and run methods.
- Load the frozen graph and create a session.

### Step 3: Create Helper Functions
- `create_label_colormap`
- `label_to_color_image`
- `vis_segmentation`

![Screenshot](https://github.com/kundamnikhil/CNST-MIT-Drive-Segmentation/assets/43941418/f61c81ae-bcf2-4175-b23a-ecb93bb183e2)

### Step 4: Load the Pre-trained Model
- Download and extract the pre-trained model.

### Step 5: Run the Model on Sample Images
- Model gives 3 images: input image, segmentation map, and segmentation overlay.

### Step 6: Evaluate the Model
- Evaluate on unknown data using pixel accuracy and mean class IoU metrics.
- Evaluate on sample image and video.

Overall, this project demonstrates the use of deep learning techniques for roadside vegetation detection from dashcam images, leveraging pre-trained models and semantic segmentation. It provides insights into model evaluation and performance metrics.
