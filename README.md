# Purpose
This program detects duplicated pictures by using a neural code from a deep network

# Research about image retrieval
Recently, outputs from a layer in a deep network are studied and those outputs, neural code, were found to be a good representaion of pictures.
[1] studied neural code of googlenet and oxfordnet. In short, [1] utilized outputs from a convolutional layer and applyed intra-VLAD [2].

# How to use
python download_model.py googlenet  
python ddp.py directory1 directory2 directory3  

you can take any numer of directories that contain pictures.



# References
[1] Exploiting Local Features from Deep Networks for Image Retrieval  
[2] All about VLAD  
