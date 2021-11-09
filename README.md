# ImageNetSplitter

### This is still work in progress!

This example is based on the pytorch example for imagenet [here](https://github.com/pytorch/examples/tree/master/imagenet)

## Requirements
- Install the requirements: `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/.  
  If you are using the aalto triton cluster, the dataset is stored at `/scratch/shareddata/dldata/imagenet/ILSVRC2012_img_train.tar` and a 
  preprocessed version is stored at `/scratch/shareddata/dldata/imagenet/imagenet21k_resized.tar.gz`
  the latter version is processed according to [this](https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_script.sh) script.
  

# Working with imagenet and other large datasets on Triton

In this tutorial we will provide information to help you work with large datasets on the triton cluster. The general information contained in this tutorial is not restricted to triton, but should be applicable to any cluster system (the numbers given of course refer to triton as of Oct 2021). We will further provide some tools and instructions particularily for the imagenet dataset.

## General Remarks
When using large datasets (>100GB) on triton, there are a few things to keep in mind:
- When transferring the dataset to a node, always try to transfer large (>3GB) files at a time. Small files will slow down your transfer speed significantly (see the section on small files).
- Network IO should always be sequential reads since the network file system is relying on spinning discs, and their IO performance is best when reading sequentially. This is the same issue as with small files.
- Loading from the network can be very efficient (the cluster does have a theoretical maximum of about 30GB/s) however, the total bandwidth is shared between all users, so constantly re-reading a dataset from the network should be avoided.
- There are nodes with SSD-raids (the `dgx` nodes), which can offer individual IO speeds from the local filesystem (i.e. `/tmp`) of up to 30GB/s. So, if a dataset needs to be read multiple times (e.g. multiple retrainings), and doesn't fit into memory, using one of these nodes, transferring the data to the node and loading from the local drive will be as efficient as it gets.
- For individual runs (i.e. reading the data once), the most efficient way is reading directly from the network. 
- Sharding (see below) the dataset, can help to further increase I/O efficiency, and potentially avoid randomisation issues.



## Sharding
Sharding a dataset means to split a huge dataset into large subsets (3-10GB/shard). This representation allows the following benefits:
1. Multiple processors can read in different Shards simultaneously, potentially increasing read efficiency (on Network drives).
2. Improved randomisation of datasets. Since shards can be read at random, you avoid always presenting the same instances first. While this does not provide real randomisation, it is a big improvement over just sequentially reading a single huge dataset.

## Best practice with `webdataset` and `pytorch`

`webdataset` offers a convenient way to load large and sharded datasets into `pytorch`, it implements the `iterabledataset` interface of `pytorch`, and can thus be used like a `pytorch` dataset. 

### Dataset format: 
`webdataset` expects the data o be stored as a tarball (`.tar`) file, with all data for each item stored in individual, sequential files. e.g.:

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

for a dataset with image-data in `image1.png` , class information in `image1.cls` and metadata in `image1.meta`. Only the image file and the class file are necessary to train machine learning algorithms.
It is essential, that the data is stored in this order, since otherwise sequential reading of the data is impossible, which would make extremely inefficient random access necessary. The file suffixes will be interpreted as keys by `webdataset`.

Luckily, the `tar` program offers you to sort a tar file by name (`--sort=name`). Make sure, that there are no items with identical names in your dataset!

### Creating Shards

The authors of `webdataset` offer a convenience tool `tarp` that allows you to easily create a sharded tar files from a folder with your dataset (assuming your dataset is stored in tuples as mentioned above). The tarp command is webdataset specific and can be installed according to the instructions on:
https://github.com/webdataset/tarp

On a dataset folder, you can run:
`tar --sorted -cf - . | tarp cat --shuffle=5000 - -o - | tarp split - -o ../dataset-%06.tar`
which will create a sharded dataset with default maximium file sizes (3 GB) and maximum number of files (10.000). This dataset is "shuffled" with a buffer of 5000 elements, i.e. if your original order is 10.000 Images of class A, 10.000 images of class B..., you will start with at least 5.000 class A images and only then slowly moving to class B,, so your final dataset will still be very "unshuffled".


### ImageNet

ImageNet commonly has a split metadata <-> image-files, so this needs to be put together seperately. `webDataset` offers a map function that can be used for this see e.g. at 4:42 in [this](https://www.youtube.com/watch?v=v_PacO-3OGQ) video.
For data that is split between a tar of tars (e.g. the train)



### Sharding programmatically
One issue that can come up is naming of the files in imageNet. If they start with their WordNet-ID sorting them might lead to them being ordered according to their WNID, which can lead to shards containing only one class.
In order to avoid this, a one-time preprocessing step for imagenet datasets is to move them into shuffled shards. Since doing this "in place" is impossible (the datasets need too much memory), the files need to be extracted from the imagenet file once. 
In order to do this on Triton, use a node with a large SSD (`srun -p interactive -gres=spindle` ) and copy the dataset you want to Shard to the /tmp folder.
Extract it there and build the image to ID mapping.
Then run `makeShards` providing the mapping and the imagefile names.


### 


