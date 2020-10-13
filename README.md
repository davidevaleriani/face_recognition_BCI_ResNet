# face_recognition_BCI_ResNet
Data and Python scripts accompanying PLoS ONE paper **[Cyborg groups enhance face recognition in crowded environments](https://doi.org/10.1371/journal.pone.0212935)**.

The main experimental script is *exp.py*, and the protocol of the experiment is described in the paper.

The folder *imagesBW* contains the black-and-white images (with normalized grayscale) used for the paper. However, these were created from original coloured images, which are in the folder *images*.

The script *automatic_face_recognition.py* uses the [face_recognition library](https://github.com/ageitgey/face_recognition) and pre-trained ResNet model to
automatically classify each image, and also estimate the confidence of the algorithm.
