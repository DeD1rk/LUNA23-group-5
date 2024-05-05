# LUNA23 group 5
# Handige linkjes :)
1. https://pubs.rsna.org/doi/10.1148/radiol.223308
"Prior CT Improves Deep Learning for Malignancy Risk Estimation of Screening-detected Pulmonary Nodules"

2. https://pubs.rsna.org/doi/epdf/10.1148/radiol.2021204433
"Deep Learning for Malignancy Risk Estimation of Pulmonary Nodules Detected at Low-Dose Screening CT"
They only did malignancy prediction. They used 9 2D slices and an existing 3D model (usually used on videos) to classify the images. 

3. https://www.sciencedirect.com/science/article/pii/S1746809422002233#s0020
"Benign-malignant classification of pulmonary nodule with deep feature optimization framework"
Utilizing a deep feature optimization framework DFOF to enhance bening-malignant classification. This is achieved by spliting the network in 2 streams. One searching for intranodular and perinodular features and the other for revealing intranodular information.

4. https://www.sciencedirect.com/science/article/pii/S0895611121000343#sec0010 " On the performance of lung nodule detection, segmentation and classification" (2021). Gives a review of many algortihms and their accuracies. Small thing about multi task learning and that it is effective. Probs mostly relevant for when we are writing the paper. 


https://github.com/hassony2/kinetics_i3d_pytorch
aformentioned 3D model used in the 2nd paper

Multi task learning papers:

1. https://github.com/uci-cbcl/NoduleNet,https://link.springer.com/chapter/10.1007/978-3-030-32226-7_30 . "First, because of the mismatched goals of localization and classification, it may be sub-optimal if these two tasks are performed using the same feature map. Second, a large receptive field may integrate irrelevant information from other parts of the image, which may negatively affect and confuse the classification of nodules, especially small ones." Made a MTL and describe their choices, code for the network can be found at the github.
2. https://github.com/CaptainWilliam/MTMR-NET, https://ieeexplore.ieee.org/document/8794587 (hiervan lukt t me om een of andere reden niet de paper daadwerkelijk te openen), maar wel code van een MLT i guess
 
