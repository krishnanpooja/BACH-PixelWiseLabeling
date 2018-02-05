# pytorch-semseg

## Semantic Segmentation Implemented in PyTorch

Code to perform pixel-wise labeling of whole-slide images. 


<p align="center">
<a href="https://www.youtube.com/watch?v=iXh9aCK3ubs" target="_blank"><img src="https://i.imgur.com/agvJOPF.gif" width="364"/></a>
<img src="https://meetshah1995.github.io/images/blog/ss/ptsemseg.png" width="49%"/>
</p>


### Networks implemented

* [FCN](https://arxiv.org/abs/1411.4038) - All 3 (FCN8s) stream variants

#### Upcoming 

* [U-Net](https://arxiv.org/abs/1505.04597) - With optional deconvolution and batchnorm


### Dataset
* [BACH](https://iciar2018-challenge.grand-challenge.org/)

### Requirements

* pytorch >=0.3.0
* torchvision ==0.2.0
* visdom >=1.0.1 (for loss and results visualization)
* scipy
* tqdm

#### One-line installation
    
`pip install -r requirements.txt`


**To train the model :**

```
python train.py [-h] [--img_rows [IMG_ROWS]] [--img_cols [IMG_COLS]]
                [--n_epoch [N_EPOCH]] [--batch_size [BATCH_SIZE]]
                [--l_rate [L_RATE]] 

  --img_rows       Height of the input image
  --img_cols       Width of the input image
  --n_epoch        # of the epochs
  --batch_size     Batch Size
  --l_rate         Learning Rate
```

