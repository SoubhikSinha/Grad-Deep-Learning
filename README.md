# Grad Deep Learning

Hello Everyone !

Welcome to my <b>`Deep Learning`</b> repository.

> <i>**Deep Learning** : Because sometimes, teaching machines to think is easier than explaining things to humans. Today, we're not just training models—we're training the next generation of overachieving robots !</i>

<br>
The above might sound a bit ironic — some parts might feel like a tall tale, but the rest ? Well, that's just <i>"The Reality"</i> !
<br>
<br>

In this repository, you'll find practical implementations of various deep learning concepts, demonstrating how the theoretical components come to ***life in action***.
<br>
> NOTE : The programming language used here (i.e., in every jupyter notebook) is **Python**, as it is the most widely used in the industry.
> 
<br>
<br>

### **[Data Analysis, ML Models and PyTorch](https://github.com/sricks404/Grad-Deep-Learning/blob/main/Data%20Analysis.%20ML%20Models%20and%20PyTorch.ipynb)** 👇<br>

This notebook acts as a referesher to your Machine Learning concepts. Later in this notebook, you will find a practical use case of **[PyTorch](https://pytorch.org/)** - which is a popular Deep Learning framework, used widely among researchers, academic institutions, students and leading companies in the tech industry. 
<br>
<br>
This notebook is divided into three steps, each containing unique tasks (sub-steps). We'll go through these sub-steps one by one to help you quickly refresh your Data Analysis and Machine Learning concepts. However, I strongly recommend that you go through the entire notebook and skim through the code cells as well🔽
<br>
> **NOTE:** If you're completely new to Deep Learning, no worries! I've created a separate repository, **[Grad-Machine-Learning](https://github.com/sricks404/Grad-Machine-Learning)**, which can help you get started with Machine Learning to have a deep practical understanding of the subject. Once you've gone through that, you'll feel confident enough to dive into the world of Deep Learning.

<br>

 1. **Data Analysis and Pre-processing**🔻
	 
	 A. Selecting a Real-World Dataset
	   > You are free to experiment with any dataset you want apart from the ones listed in the notebook.  
	   
	<br>   
	B.  Providing <b>Main Statistics</b> about the dataset (e.g. number of entries, features, etc.)
	<br>
	<br>
	C. Handling Missing Entries (Possible Techniques to apply are listed below)🔻<br>
	 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <i><b>C.1.</b> Drop rows with missing entries (If you have a large dataset and only a few missing features, it may be acceptable to drop the rows containing missing values)<br>
	 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <b>C.2.</b> Impute missing data. Replace the missing entries with the mean/median/mode of feature (You can use K-Nearest Neighbor algorithm to find the matching sample.)</i>
	<br>
	<br>
	D. Handling mismatched string formats (if any)
	<br>
	<br>
	E. Handling Outliers (if any) - (Possible Techniques to remove outliers are listed below)🔻<br>
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <i><b>E.1.</b> Eliminating Rows containing the outliers (<b>iff</b>  if the no. of outliers are limited)<br>
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <b>E.2.</b> Imputing outliers - Replacing the outliers with the mean/median/mode of the feature.</i>
	<br>
	<br>
	F. Data Visualization - Understand what patterns and information is hidden inside the data.
	<br>
	<br>
	G. Identify uncorrelated or unrelated features - computing <i>Correlation Matrix</i> between independent and target features.
	<br>
	<br>
	H. Converting Feature Values ▶️ String Datatype <b>TO</b> Categorical (Possible techniques are listed below)🔻<br>
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <i><b>H.1.</b> One-Hot Encoding (OneHotEncoder can be found in <u>scikit-learn</u> library)<br>
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <b>H.2.</b> </i> Label Encoding
	<br>
	<br>
	I. Normalizing Non-Categorical features.
	<br>
	<br>
	J.  Chooosing Target Feature (<b>y</b>) and Independent Feature(s).(<b>X</b>)
	<br>
	<br>
	K. Splitting the datasets in training, validation and testing sets (You can make use of train_test_split() function from scikit-learn OR may do it manually. The usual split ratio can be ▶️ train : validation : test = <b>80 : 10 : 10</b> or <b>70 : 15 : 15</b>).
	<br>
	<br>
	L. Confirming the shapes of X_train, X_test, y_train and y_test after performing "splitting".

<br>

 2. **ML Models**🔻
 
     A. Applying ML Algorithms
     <br> 
     
     B. Providing comparison results on different ML Models (via graph representation and appropriate reasoning about the results obtained).
     
<br>

 3. **Introduction to PyTorch and Building a Neural Network (NN)** 🔻
 
	  Before we dive into the content covered in this step, let us first have look about - **What is PyTorch ?** <br>
	  
	 > NVIDIA says that, **PyTorch is a versatile Python-based framework for creating complex machine learning models, particularly those used in image and text analysis.** Its strength lies in its ability to rapidly develop and test models, thanks to its compatibility with powerful GPUs and its dynamic approach to building and adjusting neural networks.
	
	<br> 
	
	The official PyTorch website has numerous sections to explore to get started. It also offers a 60 minutes course by the name : **[Deep Learning with PyTorch : A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)**. You need to go through the 4 mentioned notebooks on the page (each one is on a unique topic) to understand how deep learning is performed using PyTorch.
	<br>
	<br>
	
	After completion of the above course, you will be encountering the below steps (mentioned / carried out in this notebook)🔻<br>
	 - Building a shallow Neural Network (on the problem statement defined earlier) - covered in step 1 and step 2.
	 - Saving the shallow NN model.
	 - Comparing and Analyzing the results obtained.

<br>
<br>
<br>

### **[OCTMNIST Classification](https://github.com/sricks404/Grad-Deep-Learning/blob/main/OCTMNIST%20Classification.ipynb)** 👇<br>

***[MedMNIST](https://medmnist.com/)*** is a collection of standardized biomedical image datasets designed for benchmarking machine learning algorithms in medical image analysis. It includes a diverse range of medical imaging modalities, such as X-rays, MRIs, and histopathology images, covering various medical conditions. The datasets are preprocessed and resized to uniform resolutions, making them easy to use without extensive preprocessing. MedMNIST offers a standardized benchmark for evaluating the performance of machine learning models, facilitating consistent comparisons across studies. As an open-source resource, it is freely available for academic and research purposes, contributing to the advancement of AI in healthcare by providing accessible datasets for developing and testing machine learning models.
<br>
<br>
***OCTMNIST*** is a dataset that is part of the MedMNIST collection, specifically focused on **[Optical Coherence Tomography (OCT)](https://www.aao.org/eye-health/treatments/what-is-optical-coherence-tomography)** images of the retina. OCT is a non-invasive imaging technique widely used in ophthalmology to capture detailed cross-sectional images of the retina, helping in the diagnosis and monitoring of various eye conditions, such as macular degeneration and diabetic retinopathy.<br>
**Key Features**🔻

-   **Medical Imaging Modality** : The dataset consists of OCT images, which are used to examine the retina's structure and identify abnormalities.
-   **Preprocessed Images** : Like other datasets in the MedMNIST collection, OCTMNIST images are preprocessed and resized to a standard resolution (typically 28x28 pixels), making them easier to use in machine learning models without extensive preprocessing.
-   **Classification Task** : The primary task associated with OCTMNIST is classification, where the goal is to classify the images into different categories based on the presence or absence of retinal conditions.
-   **Benchmarking** : OCTMNIST provides a benchmark for evaluating the performance of machine learning models on OCT image classification, facilitating consistent comparisons across different algorithms.

<br>

This notebook will demonstrate the implementation of a Neural Network using **PyTorch** for OCTMNIST Classification problem. Below are the steps carried out for the aforementioned 🔽

 - Downloading OCTMNIST 2D Dataset and Preparing it for Training
	 - Preprocessing steps ▶️ Normalizing Pixel Values to a Standardized Range (between 0 and 1)
	 - Train : Validation : Test (split ratio) = **70 : 15 : 15**
	 <br>
 - Building a Neural Network
	- Including Convolutional Layers and Fully Connected (FC) Layers
	- Introduction of Activation Functions after each layer to introduce *"Non-Linearity"*
		> NOTE🔻 
			1. The model was trained on GPU. You can do the same on your local system if you have GPU Installed in it (e.g. NVIDIA GTX 1050 on a Windows Machine).
			2. If you want to do so, you may follow the tutorial :   ***["How to setup CUDA GPU for PyTorch on a Windows Machine"](https://www.youtube.com/watch?v=r7Am-ZGMef8&pp=ygUTQ1VEQSBHUFUgd2luZG93cyAxMQ%3D%3D)***
			3. It's completely alright if you do not want to take the hassle of setting up GPU for PyTorch on your local system (like me !). There are alternatives🔻<br>
			&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A. [Google Colab](https://colab.research.google.com/) has numerous GPU / computation accelerator options. [**LINK**](https://www.geeksforgeeks.org/how-to-use-gpu-in-google-colab/#)<br>
			&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;B. [Kaggle](https://www.kaggle.com/) is also an option to explore !!! [**LINK**](https://www.kaggle.com/code)

<br>

 - Applying Below Techniques to Prevent Overfitting and Enhance Model Performance🔻
	 - [Regularization](https://www.ibm.com/topics/regularization#:~:text=Regularization%20is%20a%20set%20of,overfitting%20in%20machine%20learning%20models.) (L1/L2)
	 - Introduction of [Drop-Out Layer(s)](https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9) (between Fully Connected Layers)
	 - [Early Stopping](https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/)

<br>

 - Saving the weights of the trained Neural Network that provided the best results
 
 <br>
 
 - Discussing the results and providing relevant graphs for comparison and analysis🔻
	 - Analysis of Evaluation metrics like - Training Accuracy, Training Loss, Validation Accuracy, Validation Loss, Testing Accuracy and Testing Loss
	 - Plotting (per epoch) - Training **vs** Validation Accuracy
	 - Plotting (per epoch) - Training **vs** Validation Loss
	 - Confusion Matrix on mode's prediction of test set
	 - Model Performace report on other metrics : Precision, Recall (Sensitivity) and F1-Score.

<br>
<br>
<br>

### **[CNN Classification (VGGNet)]()** 👇<br>

Before diving into the contents of the notebook, let's cover some theoretical aspects of CNNs (Convolutional Neural Networks)🔻<br>
> A **Convolutional Neural Network (CNN)** is a type of deep learning model designed primarily for processing structured grid data like images. CNNs work by automatically and adaptively learning spatial hierarchies of features through the application of convolutional operations. These networks consist of layers that apply filters (kernels) to input data, which helps in detecting patterns like edges, textures, and more complex shapes in deeper layers. Key components include convolutional layers, pooling layers (for downsampling), and fully connected layers (for classification). CNNs are widely used in computer vision tasks such as image classification, object detection, and facial recognition due to their ability to capture spatial relationships in data efficiently.

<br>

There are different architectures of CNN. In this notebook, you will particularly oberve the implementation of **[VGGNet](https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/)** 🔻<br>
> VGGNet is a convolutional neural network architecture introduced by the Visual Geometry Group at the University of Oxford in 2014, known for its simplicity and effectiveness in image recognition tasks. The architecture is characterized by its uniform use of small 3x3 convolutional filters across all layers, allowing it to capture complex features while maintaining computational efficiency. VGGNet is deep, with popular versions like **VGG16 and VGG19** containing **16 and 19 layers**, respectively, alternating between convolutional and max-pooling layers. This deep structure, combined with fully connected layers at the end, enables VGGNet to achieve high accuracy in image classification tasks. However, its depth and the use of fully connected layers lead to high computational costs and a large number of parameters, making it memory-intensive and prone to overfitting, especially on smaller datasets. Despite these limitations, VGGNet remains a foundational model in deep learning, serving as a baseline for many subsequent architectures.

<br>

If you want to check out the architectures of different VGG versions, you can refer to this research paper : **[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)**
<br>
<br>

Now, let's see the steps taken for the implementation of the VGG-13 version🔻<br>

 - Data Preprocessing🔻
	 - Reading data
	 - Preprocessing data
	 - Preprocessing steps🔻
		 - Image Normalization
		 - Label Encoding & One-Hot Encoding
	- Main Statistics🔻
		- Mean
		- Standard Deviation
		- Data Distribution
	- Data Visualization

<br>

- VGG-13 (Version B) implementation

<br>

- Model Training

<br>

-  Applying Hyperparameter Tuning techniques🔻
	- Regularization (L1/L2)
	- Dropout
	- Early Stopping
	- Image Augmentation

<br>

- Saving the weights of the model which gave the best results

<br>

- Discussing the results and providing relevant graphs for comparison and analysis🔻
	 - Analysis of Evaluation metrics like - Training Accuracy, Training Loss, Validation Accuracy, Validation Loss, Testing Accuracy and Testing Loss
	 - Plotting (per epoch) - Training **vs** Validation Accuracy
	 - Plotting (per epoch) - Training **vs** Validation Loss
	 - Confusion Matrix on mode's prediction of test set
	 - Model Performace report on other metrics : Precision, Recall (Sensitivity) and F1-Score.

<br>
<br>
<br>

# ⚠️ README.md under construction
