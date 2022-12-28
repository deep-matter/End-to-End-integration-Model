# End-to-End-integration-Model
this project was collboration with **The national university of water and Environmental Engineering** to develop the web application to diagnosis Eyes diseaes Retina Pathway , and it was Phd Philosophy under title ***Development Deep Neural Network for diagnostics Eyes diseaes***
1. Description Project:
   - we worked on End-to-End integration Models that contain tWo models **Classification** and **Segementation** each one of it built using different technologies , but important step in any Machine Learning Process is Deployment Model into production , we developed a **Web Application** using Flask to creat two EndPoints REST APIs . 

### Build the models:
as follows we built two models classifcation and segmentation and each one asychronized way. 

* **Classification** :
    - problem description :
      at this stage we explored different models architecture in computer vision , and we used Convolution neural network with adding improvement to it 
      1. collecting data :
         we obtained the data from **Kaggle** and the team Lab Labeled it manually with expeter and it has Four Classes (DiabeticRetinopathy , Glaucoma , Myopia , Normal)
      2. Processing :
         the data was nosiy and we used Per-Processing technique to enhance data such 
         super Resolution (Autoencoder) , Oversampling (DCGAN) , Histogram Equilazation 
      3. built model :
         we explored different models architecture such RESNET and MobileNEt from Tensoflow **Per-Trained** models and we funed tune the parameters we reach out wiht out final model to ***96.58% AUC*** and it was based CNN and Smoothing is technic Regulazition to vectorize labels , and all the training process was implemented on Colab Google Platfrom that Provide GPU
* **Segmentation** :
    1. Problem Description:
      after we built the Classifier and test on different sample some problems we faced are :
      - similar features between Classes which make the model not able to predict each class good , 
      - **Map vessel blood** of eyes have low pixles range and hard to make model focus on **the region of interest**. in this stage we used ***UNET MODEL***  to Re-Create a new database which only have the Map Vessel Boold , 
    2. collecing data  




