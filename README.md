# Anime Character Face Generation model using DCGAN
- The Anime Character Face Generation model uses Deep Convolutional Generative Adversarial Networks (DCGANs) to create realistic anime character faces. DCGANs are a type of GAN (Generative Adversarial Network) that uses deep convolutional networks to improve the stability and quality of generated images. By training on a dataset of anime faces, the model learns the intricate details and patterns that define anime characters, enabling it to generate new, unique faces that closely resemble the training data.
- Model uses unsupervised learning which does not require labeled data for training processes.
- Without worrying about copyright issues anime artists can make use of the generator to get inspiration to create new characters.
### Datasets
- Combining the below 2 datasets makes thee training dataset. Total of **85,117** images.
- Dataset1: https://www.kaggle.com/datasets/splcher/animefacedataset
- Dataset2: https://www.kaggle.com/datasets/soumikrakshit/anime-faces
## Architecture
![image](https://github.com/darsini-k22/dcgan-anime-generation/assets/75623259/889c23b1-b95a-4f07-b688-dbd75ff67cef)

### Installations
- Python == 3.11.9
### Used Libraries
```
tensorflow == 1.12.0
tqdm == 4.66.1
keras == 3.0.5
matplotlib == 3.8.3
numpy == 1.26.4
```
### Imports
```
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img, array_to_img
from keras.models import Sequential, Model
from keras import layers
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import random
warnings.filterwarnings('ignore')
```

### At the 50th epoch the output sample is shown below
![image](https://github.com/darsini-k22/dcgan-anime-generation/assets/75623259/143a2611-5aef-46e3-b843-d1b1a535072f)


