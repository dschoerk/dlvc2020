To recreate the Conda environment run the following commands. The EfficientDet code has been tested on Windows 10 with RTX 2070 and RTX 2080Ti.

`conda env create -f environment.yml`

`conda activate efficientdet-pytorch`

`pip install -r requirements.txt`

`python efficientdet_mytest.py`

If errors are popping up during installation it is most likely due to missing packages. You can install them manually. 