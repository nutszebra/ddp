# Purpose
This program detects duplicated pictures by using a neural code from a deep network

# Research about image retrieval
Recently, outputs from a layer in a deep network are studied and those outputs, neural code, were found to be a good representaion of pictures.
[1] studied neural code of googlenet and oxfordnet. In short, [1] utilized outputs from a convolutional layer and applyed intra-VLAD [2].

# How to use

    git clone https://github.com/nutszebra/ddp.git
    cd ddp
    ipython
    run download_model.py googlenet  
    run ddp.py directory1 directory2 directory3  

You can give some directories that contain pictures. Duplicated pictures are stored in dup variable:

    dup
    
Neural code of each pictures is stored into feature. feature is dictionary type, thus you can check neural code like this:

    feature["directory1/a.jpg"]

There is a method to search near pictures. I search for near pictures of directory1/a.jpg:

    searchNearPic("directory1/a.jpg", 0.5, feature)

Second argument is threshold of cosine distance.

# References
[1] Exploiting Local Features from Deep Networks for Image Retrieval  
[2] All about VLAD  
