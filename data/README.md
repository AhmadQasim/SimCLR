# For adding support for new datasets

### Organized dataset
If the dataset is an image classification and is organized in the following format: 
```
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
```

Use the ImageFolder class for returning the train and test datasets (https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder)

**Note: View jurkat_dataset, matek_dataset or plasmodium_dataset as an example**

### Unorganized dataset
If the dataset is not organized in the previously mentioned format, then add a CustomDataset (https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) class wrapper and return the train and test datasets.

## Loading dataset
Add a conditional statement in main.py to load the respective train and test datasets