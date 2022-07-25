# MFEIF_Lite

### Install
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
### File structure
```
  ├── Image_Fusion
  |   ├── train.py
  |   ├── fuse.py
  |   └── ...
  └── datasets
      ├── FLIR_ADAS_v2
      |   ├── images_rgb_train
      |   |   └── data
      |   |       └── ...
      |   ├── images_rgb_val
      |   |   └── data
      |   |       └── ...
      |   ├── images_thermal_train
      |   |   └── data
      |   |       └── ...
      |   ├── images_thermal_val
      |   |   └── data
      |   |       └── ...
      └── ...
```

### Train
```
python train.py
```

### Fuse
```
python fuse.py
```