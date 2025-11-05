#### **Solution explanation**



1. ###### Dataset preparation

First of all dataset from [Kaggle](https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine) was downloaded.



Then we read geojson file with geopandas library. We also make sure that we have the same coordinate system in our images and dataframe (so we transform coordinate system of dataframe to coordinate system of images). Then we divide all Multipolygons into multiple Polygons in geometry column of dataframe by using explode() function. And also we set bounds of our polygons, it will be much easier to work with rectangle when cropping images.



Next step is cleaning dataframe. Now it has many rows with dates that are absent in our dataset. After cleaning we remain with 1483 rows instead of ~5600 rows. I also removed folders with dates that are absent in our dataframe, but that step is not necessary at all.



Next step is calculating area of polygons and looking at common percentiles. We see a lot of very small polygons (some are smaller than 1 pixel). We don't need such small polygons so we remove all polygons where area is lower than 500. We remained with 1244 rows in our dataframe.



Next step is to found all polygons on the same tile that intersects (with a little buffer because surroundings are important too and deforestation region can move a little bit through time). We make sure we take different polygons (different dates of images) and now we have 2582 positive pairs.



Then we crop 32x32 parts from those images, where polygon is located. For this purpose we created function that find paths of images by tile\_id and img\_date. We crop polygon from B02 (blue), B03 (green) and B04 (red) bands and by stacking them we have an rgb image 32x32. Doing this for each pair we got in previous steps we have 2300 pairs generated for dataset (some image crops contained only zeroes because some images weren't perfect). We give label 1 for those pairs (images are matching).



Then for dataset balance we must create negative pairs (different polygons). For this purpose we divide dataset into two with different tiles (locations) so we don't get pair matching accidentaly. We are getting subsample of length 120 from each part of dataset. And then generate all possible pairs for first 40 rows of datasets, second 40 rows of datasets and third 40 rows of dataset. So we get 4800 potential pairs.



Then we crop 32x32 parts from those images, where polygon is located. We crop polygon from B02 (blue), B03 (green) and B04 (red) bands and by stacking them we have an rgb image 32x32. Doing this for each pair we got in previous steps we have 3613 pairs generated for dataset (some image crops contained only zeroes because some images weren't perfect or other errors occured). We give label 0 for those pairs (images aren't matching). So we have 5913 pairs total.



Lastly we divide dataset into training, test and validation parts (80/10/10) and saving them as .npz files (format for numpy array).





###### 2\. Selecting model architecture

For image matching Siamese model was chosen. It consists of two CNN, that share the same structure and weights. They convert images into vectors (embeddings) and after that distance between those vectors is computed. If distance is lower than some threashold, images aren't matching. If distance is bigger than this threshols, images are matching.



CNN model has next structure:

It takes 3x32x32 image (32x32 image in rgb). It passes this image into convolutaional layer with kernel\_size = 3. After that we have 32 feature maps. We normalize them, use activation function (Relu). Then reduce sizes of those maps to 16x16 by using pooling layer.

Next those 32 feature maps are passed into another convolutaional layer with kernel\_size = 3. After that we have 64 feature maps. We normalize them, use activation function (Relu). Then reduce sizes of those maps to 8x8 by using pooling layer.

Next those 64 feature maps are passed into another convolutaional layer with kernel\_size = 3. After that we have 128 feature maps. We normalize them, use activation function (Relu). Then we reduce sizes of those maps to 1x1 by using adaptive pooling layer.

Then we have full-connected layer. First layer has 128 neurons as input and 256 neurons as output. We apply activation function (Relu). Next we have dropout, which is working only while model is training. It turns off 25% randomly chosen neurons (their output is zero). This helps to get good weight on all neurons and not depending on only few. Lastly we have layer that takes 256 neurons as input and 64 neurons as output. Those 64 neurons are our embedding vector, final output of model.





###### 3\. Training model

First we load train and validation data, that we prepared and saved earlier. We transform it into more appropriate for torch format and also create DataLoader that will give images by batches for faster learning. We will use GPU if possible (if not we can use CPU but it will be slower).



Also we made loss function for our task. If images are matching (label = 1) we punish model if distance between embeddings is too big. If images aren't matching (label = 0) we punish model if distance between embeddings is too low. We take euclidean distance for this task.



We setted next parameters for training:

Learning rate is 1e-3 and is decreased by half every 10 steps (epochs). By this our model will adjust its weights more carefully in last epochs and will not throw away what it already learned. Weight decay is equal to 1e-4. This helps preventing overfitting model. Very important parameter on small datasets. Number of epochs is setted to 50. By this model will be on time to learn data patterns, but won't overfit too much.



Then we train our Siamese model by common pattern for each neural network:

In every epoch we give our model pairs of images from training dataset, getting distance between embeddings. Then we calculate loss function, do backward propagation and update our weights. After we finish one epoch we give our model pairs of images from validation dataset, getting distance between embeddings and calculate loss function. By that we can see if our model is learning. If that validation loss is lowest so far, we save that model. We also print train loss and validation loss after each epoch ended to see how our model is training.





###### 4\. Testing model

First we load test data, that we prepared and saved earlier. We transform it into more appropriate for torch format and also create DataLoader that will give images by batches for faster evaluation. We will use GPU if possible (if not we can use CPU but it will be slower).



We also load model, that we trained and saved earlier.



We give all pairs of images from test dataset to model, getting distance between embeddings. Then we calculate loss function. We also set threshold as 0.5. If distance is lower than this threshold, we consider images are matching. Otherwise images are not matching. By setting this threshold we also can calculate accuracy of our model.





###### 5\. Preparing demo

Demo starts with explanation of what it does and what it contains.



Then it lists all libraries that are needed to run demo and command that will install them if they aren't installed already.



Then after importing all necessary libraries. We are loading our model and loading test data. We will use GPU if possible (if not we can use CPU but it will be slower).



First we evaluate test metrics like we did in previous chapter. We also computed precision, recall and F-score additionally.



After that we take two pictures of the same tile in different times, crop 2048x2048 area from each image at the same place. We find some keypoints on each image by using cv2 library. Then we take 32x32 crops for each keypoint and give it to our model, getting distance between embeddings. Best matches are shown on visualization.







#### Details on how to set up project.

If you want just to use this model you only need to download demo, training\_model.py and best\_siamese\_model.pth file. In demo there is example how to use model.



If you want to repeat process of creating that model, you need to follow instructions:

1. First of all you to need to download dataset from [Kaggle](https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine).
2. After that you need to download all .py and .ipynb files.
3. Assuming you already have Python 3.12 (On moment of writing this readme file, torch doesn't support using GPU for faster training and predictions on Python 3.13, so I needed to downgrade to Python 3.12) you need to download all libraries listed in requirements.txt. You can do by writing next command in console: pip install 'library\_name'.
4. To use GPU for faster training and predictions your GPU must support CUDA (If you have NVIDIA GPU you need to check version of CUDA it supports. Then you can import torch and by using torch.cuda.is\_available() function check. If it is False you might need to run next command (replace 121 with your version of CUDA if necessary): 'pip install torch==2.2.2+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121').
5. If you have not NVIDIA GPU or torch.cuda.is\_available() says False no matter what you do, it is okay. It will run by CPU, it is much slower, but it must work. If torch.cuda.is\_available() says True, you can go to next step.
6. Open dataset\_generation.ipynb and run all cells. At the end three .npz files (pairs\_train\_data.npz, pairs\_test\_data.npz, pairs\_validation\_data.npz) must appear in your directory. In cell where number of pairs in dataset is printed you must have 5913.
7. Next step is to open model\_training.py file and run it. You must see train\_loss and validation\_loss printed every epoch. If validation\_loss is best so far model will be saved. For me training model took about 5 minutes. After training will end, you must see best\_siamese\_model.pth file appear in your directory.
8. Now after you have trained you model, you can run model\_testing.py or demo.ipynb to check result you've got.
