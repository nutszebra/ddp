# purpose
This program detects duplicate pictures by using neural code

# how
According to [1], an output from top layer of a deep learning network has a good representation of input data.
By using the fact, [2] used a neural code from alexnet to eliminate duplicate and near-duplicate images.
This program obtains a neural code from alexnet(fc7) and detects duplicate pictures.
Chainer and modelZoo is used.

# references
[1] BABENKO, A., SLESAREV, A., CHIGORIN, A., AND LEMPITSKY, V. S. 2014. Neural codes for image retrieval. In ECCV.
[2] BELL, S., AND BALA, K. 2015. Learning Visual Similarity for Product Design with Convolutional Neural Networks. ACM TOG 34, 4.
