# ImageNetTools

### This is still work in progress!

## Overview
This example is based on the pytorch example for imagenet [here](https://github.com/pytorch/examples/tree/master/imagenet)
It was adapted to allow it also being used for other image based datasets by adding a class count variable to the input
arguments. For other datasets or specific needs it is likely necessary to modify the transformations performed on the
images (located at `ImageNetTools/imageNetTransformation.py`), and you likely need to set the --classes parameter, to
indicate the number of classes in the dataset used.
This Explanation is divided into two parts. First there are some general remarks about large dataset on triton (and in
general any cluster based distributed file system, where files are distributed on multiple disks). And then a tutorial
on how to run ml training using the imageNet dataset and pytorch on triton.

## Requirements
- Download the ImageNet dataset from http://www.image-net.org/.  
  If you are using the aalto triton cluster, and have an imagenet account, the dataset is stored at 
  `/scratch/shareddata/dldata/imagenet/ILSVRC2012_img_train.tar` and a 
  preprocessed version is stored at `/scratch/shareddata/dldata/imagenet/imagenet21k_resized.tar.gz`
  the latter version is processed according to [this](https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_script.sh)
  script. 
  
- Create a conda environment using the requirements:  
  `conda create env -f requirements.yaml`  
  if you have mamba installed used mamba (as it is much faster in resolving environments:  
  `mamba create env -f requirements.yaml`
- Activate your environment:  
  `conda activate shardProcess`  
  or, on triton (or probably most clusters):
  `source activate shardProcess`


# Working with imagenet and other large datasets on Triton

First, we will provide information to help you work with large datasets on the triton cluster. The general information
contained in this tutorial is not restricted to triton, but should be applicable to any cluster system (the numbers given
of course refer to triton as of Oct 2021). 

## General Remarks
When using large datasets (>100GB) on triton, there are a few things to keep in mind:  
- When transferring the dataset to a node, always try to transfer large (>3GB) files at a time. Small files will slow
  down your transfer speed significantly (see the section on small files).
- Network IO should always be sequential reads since the network file system is relying on spinning discs, and their IO
  performance is best when reading sequentially. This is the same issue as with small files. In addition, at least on triton,
  large files are striped and can be served from multiple disks which speeds up IO a lot. However this feature can only
  be used if the files are read sequentially as otherwise there is no "pre-reading" from multiple disks. 
- Loading from the network can be very efficient (the cluster does have a theoretical maximum of about 30GB/s) however,
  the total bandwidth is shared between all users, so constantly re-reading a dataset from the network should be avoided.
- There are nodes with SSD-raids (all of the gpu nodes do have them), which can offer individual IO speeds from the local filesystem (i.e.
  `/tmp`) of up to 30GB/s. So, if a dataset needs to be read multiple times (e.g. multiple retrainings), and doesn't fit
  into memory, consider using one of these nodes, transferring the data to the node and loading from the local drive will be as
  efficient as it gets.
- For individual runs (i.e. reading the data once), the most efficient way is reading directly from the network. 
- For datasets which do no longer fit on the SSD drive, sharding (see below) the dataset can help to increase  
  I/O efficiency and avoids randomisation issues that would ccur when a single file is used as storage.
  


## Sharding
Sharding a dataset means to split a huge dataset into large subsets (1-10GB/shard). This representation allows the 
following benefits:
1. Multiple processors can read in different Shards simultaneously, potentially increasing read efficiency (e.g. on striped Network drives).
2. Improved randomisation of huge datasets. Since shards can be read at random, you avoid always presenting the same instances
   first. While this does not provide real randomisation, it is a big improvement over just sequentially reading a
   single huge dataset.

## Sharding with [`webdataset`](https://github.com/webdataset/webdataset)

[`webdataset`](https://github.com/webdataset/webdataset) offers a convenient way to load large and sharded datasets into
`pytorch`, it implements the `iterabledataset` interface of `pytorch`, and can thus be used like any other `pytorch` dataset. 

### Dataset format: 
`webdataset` expects the data to be stored as a tarball (`.tar`) file, with all data for each item stored in individual,
sequential files. e.g.:

```
Dataset.tar
|-> image1.png
|-> image1.cls
|-> image1.meta
|-> image2.png
|-> image2.cls
|-> image2.meta
....
|-> imageXYZ.png
|-> imageXYZ.cls
|-> imageXYZ.meta
```

for a dataset with image-data in `image1.png` , class information in `image1.cls` and metadata in `image1.meta`. Only 
the image file and the class file are necessary to train machine learning algorithms. You can even pre-process data and 
store the pre-processed data into the tarball, thus avoiding costly pre-processing steps during data loading 
(e.g. resizing, cropping a central image, conversion into a tensor and normalisation). 
It is essential, that the data is stored in this order, since otherwise sequential reading of the data is impossible, 
which would make extremely inefficient random access necessary. The file suffixes will be interpreted as keys by 
`webdataset`.

Luckily, the `tar` program offers you to sort a tar file by name (`--sort=name`) when packing. However, you have to 
ensure, that there are no items with identical names in your dataset!

### Creating Shards

The authors of `webdataset` offer a convenience tool `tarp` that allows you to easily create sharded tar files from a 
folder with your dataset (assuming your dataset is stored in tuples as mentioned above). The tarp command is webdataset 
specific and can be installed according to these [instructions](https://github.com/webdataset/tarp)

On a dataset folder, you can run:  
`tar --sorted -cf - . | tarp cat --shuffle=5000 - -o - | tarp split - -o ../dataset-%06.tar`  
which will create a sharded dataset with default maximium file sizes (3 GB) and maximum number of files (10.000). This 
dataset is "shuffled" with a buffer of 5000 elements, i.e. if your original dataset contains 10.000 Images of class A, 
10.000 images of class B..., you will start with at least 5.000 class A images and only then slowly moving to class B 
etc... so your final shards (at lest the initial ones) will still be very "unshuffled".


### Build Shards for ImageNet 

ImageNet commonly has split metadata <-> image-files, so this data needs to be put together seperately. `webDataset` 
offers a map function that can be used for this see e.g. at 4:42 in [this](https://www.youtube.com/watch?v=v_PacO-3OGQ) 
video. 
While this works for a tar-packaged file, it fails with the default format in which e.g. the imageNet training data is 
stored (which is a tar of tars). This also prevents direct use of tarp to shard the dataset. 
Here we provide some tools that allow you to shard the trainig data and simultaneously randomizing it within the shards 
thus avoiding the issue mentioned above. We also provide you with a script that, based on the imagenet example from 
pytorch, allows you to train a pytorch model with the generated sharded imagenet dataset.

Sharding with full randomisation requires either the full dataset to be available on a hard drive in individual files, 
or a sufficient amount of memory to store the whole dataset at once. Since imagenet is quite large (140GB for the 
normal, 210GB for the filtered larger dataset), this can only be done in memory on a limited number of nodes, or needs a
 fast hard drive. On triton the gpu nodes all have fast SSD drives that can be used for sharding locally, i.e. you want 
 to go to the gpu or gpushort partitions. However, make sure, that you are not requesting GPU ressources, since they 
 are completely unnecessary for sharding.

The `imagenet_sharding.py` script allows you to easily create a sharded imagenet dataset. The default values fit for the
 imagenet files available on Triton. If you want to use it 

For Validation data, a sharded set is available on triton at ...

You can try running it on the bigmem queue:



