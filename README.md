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

This notebook acts as a referesher to your Machine Learning concepts. Later in this notebook, you will find a practical use case of **[PyTorch](https://pytorch.org/)** - which is a popular Deep Learning framework, used widely by researchers, academic institutions, students and leading organization in the tech industry. 
<br>
<br>
This notebook is divided into three steps, each containing unique tasks (sub-steps). We'll go through these sub-steps one by one to help you quickly refresh your Data Analysis and Machine Learning concepts. However, I strongly recommend that you go through the entire notebook and skim through the code cells as well🔽
<br>
> **NOTE :** If you're completely new to Deep Learning, no worries! I've created a separate repository, **[Grad-Machine-Learning](https://github.com/sricks404/Grad-Machine-Learning)**, which can help you get started with Machine Learning to have a deep practical understanding of the subject. Once you've gone through that, you'll feel confident enough to dive into the world of Deep Learning.

<br>

 1. **Data Analysis and Pre-processing**🔻
	 
	 A. Selecting a Real-World Dataset
	   > You are free to experiment with any dataset you want apart from the ones listed in the notebook.  
	   
	<br>   
	B.  Providing <b>Main Statistics</b> about the dataset (e.g. number of entries, features, etc.)
	<br>
	<br>
	C. Handling Missing Entries (Possible Techniques to apply are listed below)🔻<br>
	 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <i><b>C.1.</b> Dropping rows with missing entries (If you have a large dataset and only a few missing features, it may be acceptable to drop the rows containing missing values)<br>
	 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <b>C.2.</b> Imputing missing data</i> ▶️ <i>Replacing the missing entries with the mean/median/mode of feature (You can use K-Nearest Neighbor algorithm to find the matching sample.)</i>
	<br>
	<br>
	D. Handling mismatched string formats (if any)
	<br>
	<br>
	E. Handling Outliers (if any) - (Possible Techniques to remove outliers are listed below)🔻<br>
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <i><b>E.1.</b> Eliminating Rows containing the outliers (<b>iff</b>  if the no. of outliers are limited)<br>
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <b>E.2.</b> Imputing outliers ▶️ Replacing the outliers with the mean/median/mode of the feature.</i>
	<br>
	<br>
	F. Data Visualization ▶️ Understanding what patterns and information is hidden inside the data
	<br>
	<br>
	G. Identifying uncorrelated or unrelated features ▶️ Computing <i>Correlation Matrix</i> between independent and target features
	<br>
	<br>
	H. Converting Feature Values ▶️ String Datatype <b>TO</b> Categorical (Possible techniques are listed below)🔻<br>
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <i><b>H.1.</b> One-Hot Encoding (OneHotEncoder can be found in <u>scikit-learn</u> library)<br>
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <b>H.2.</b> </i> Label Encoding
	<br>
	<br>
	I. Normalizing Non-Categorical features
	<br>
	<br>
	J.  Choosing Target Feature (<b>y</b>) and Independent Feature(s) (<b>X</b>)
	<br>
	<br>
	K. Splitting the datasets in training, validation and testing sets (You can make use of <b>train_test_split() function from scikit-learn</b> OR may do it manually. The usual split ratio can be ▶️ train : validation : test = <b>80 : 10 : 10</b> or <b>70 : 15 : 15</b>).
	<br>
	<br>
	L. Confirming the shapes of X_train, X_test, y_train and y_test after performing "splitting"

<br>

 2. **ML Models**🔻
 
     A. Applying ML Algorithms
     
     B. Providing comparison results on different ML Models (via graph representation and appropriate reasoning about the results obtained)
     
<br>

 3. **Introduction to PyTorch and Building a Neural Network (NN)** 🔻
 
	  Before we dive into the content covered in this step, let us first have a look about - **What is PyTorch ?** <br>
	  
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

 - Downloading OCTMNIST 2D Dataset and Preparing it for Training 🔻
	 - Preprocessing steps ▶️ Normalizing Pixel Values to a Standardized Range (between 0 and 1)
	 - Train : Validation : Test (split ratio) = **70 : 15 : 15**

 - Building a Neural Network 🔻
	- Including Convolutional Layers and Fully Connected (FC) Layers
	- Introduction of Activation Functions after each layer to introduce *"Non-Linearity"* <br><br>
		> NOTE🔻 <br>
			1. The model was trained on GPU. You can do the same on your local system if you have GPU Installed in it (e.g. NVIDIA GTX 1050 on a Windows Machine).<br>
			2. If you want to do so, you may follow the tutorial :   ***["How to setup CUDA GPU for PyTorch on a Windows Machine"](https://www.youtube.com/watch?v=r7Am-ZGMef8&pp=ygUTQ1VEQSBHUFUgd2luZG93cyAxMQ%3D%3D)***<br>
			3. It's completely alright if you do not want to take the hassle of setting up GPU for PyTorch on your local system (like me !). There are alternatives🔻<br>
			&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A. [Google Colab](https://colab.research.google.com/) has numerous GPU / computation accelerator options. [**LINK**](https://www.geeksforgeeks.org/how-to-use-gpu-in-google-colab/#)<br>
			&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;B. [Kaggle](https://www.kaggle.com/) is also an option to explore !!! [**LINK**](https://www.kaggle.com/code)

<br>

 - Applying Below Techniques to Prevent Overfitting and Enhance Model Performance🔻
	 - [Regularization](https://www.ibm.com/topics/regularization#:~:text=Regularization%20is%20a%20set%20of,overfitting%20in%20machine%20learning%20models.) (L1/L2)
	 - Introduction of [Drop-Out Layer(s)](https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9) (between Fully Connected Layers)
	 - [Early Stopping](https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/)

 - Saving the weights of the trained Neural Network that provided the best results
 
 - Discussing the results and providing relevant graphs for comparison and analysis🔻
	 - Analysis of Evaluation metrics like - Training Accuracy, Training Loss, Validation Accuracy, Validation Loss, Testing Accuracy and Testing Loss
	 - Plotting (per epoch) - Training **vs** Validation Accuracy
	 - Plotting (per epoch) - Training **vs** Validation Loss
	 - Confusion Matrix on mode's prediction of test set
	 - Model Performace report on other metrics : Precision, Recall (Sensitivity) and F1-Score.

<br>
<br>
<br>

### **[CNN Classification (VGGNet)](https://github.com/sricks404/Grad-Deep-Learning/blob/main/CNN%20Classification%20(VGGNet).ipynb)** 👇<br>

Before diving into the contents of the notebook, let's cover some theoretical aspects of CNNs (Convolutional Neural Networks)🔻<br>
> A **Convolutional Neural Network (CNN)** is a type of deep learning model designed primarily for processing structured grid data like images. CNNs work by automatically and adaptively learning spatial hierarchies of features through the application of convolutional operations. These networks consist of layers that apply filters (kernels) to input data, which helps in detecting patterns like edges, textures, and more complex shapes in deeper layers. Key components include convolutional layers, pooling layers (for downsampling), and fully connected layers (for classification). CNNs are widely used in computer vision tasks such as image classification, object detection, and facial recognition due to their ability to capture spatial relationships in data efficiently.

<br>

There are different architectures of CNN. In this notebook, you will particularly oberve the implementation of **[VGGNet](https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/)** 🔻<br>
> VGGNet is a convolutional neural network architecture introduced by the Visual Geometry Group at the University of Oxford in 2014, known for its simplicity and effectiveness in image recognition tasks. The architecture is characterized by its uniform use of small 3x3 convolutional filters across all layers, allowing it to capture complex features while maintaining computational efficiency. VGGNet is deep, with popular versions like **VGG16 and VGG19** containing **16 and 19 layers**, respectively, alternating between convolutional and max-pooling layers. This deep structure, combined with fully connected layers at the end, enables VGGNet to achieve high accuracy in image classification tasks. However, its depth and the use of fully connected layers lead to high computational costs and a large number of parameters, making it memory-intensive and prone to overfitting, especially on smaller datasets. Despite these limitations, VGGNet remains a foundational model in deep learning, serving as a baseline for many subsequent architectures.

<br>

If you want to check out the architectures of different VGG versions, you can refer to this research paper : **[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)**
<br>
<br>

Now, let's see the steps taken for the implementation of the VGG-13 version 🔽<br>

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

- VGG-13 (Version B) implementation ([Architecture](https://www.researchgate.net/figure/Network-Architectures-a-VGG-13-b-ResNet-6_fig3_351419846#:~:text=Figure%203%20shows%20the%20Spiking,previous%20layer%20while%20retaining%20spatial))

- Model Training

-  Applying Hyperparameter Tuning techniques🔻
	- Regularization (L1/L2)
	- Dropout
	- Early Stopping
	- Image Augmentation

- Saving the weights of the model which gave the best results

- Discussing the results and providing relevant graphs for comparison and analysis🔻
	 - Analysis of Evaluation metrics like - Training Accuracy, Training Loss, Validation Accuracy, Validation Loss, Testing Accuracy and Testing Loss
	 - Plotting (per epoch) - Training **vs** Validation Accuracy
	 - Plotting (per epoch) - Training **vs** Validation Loss
	 - Confusion Matrix on mode's prediction of test set
	 - Model Performace report on other metrics : Precision, Recall (Sensitivity) and F1-Score.

<br>
<br>
<br>

### [Implementing ResNet Architecture](https://github.com/sricks404/Grad-Deep-Learning/blob/main/ResNet%20Implementation.ipynb) 👇<br>

Let us first understand a bit about **ResNet Architecture** 🔻<br>
> <b>NOTE : </b>  ResNet, or Residual Network, is a deep learning architecture that uses skip connections, allowing the network to learn residuals between layers rather than full transformations. This helps train very deep networks efficiently without vanishing gradient issues. Each "residual block" passes the input through a few layers and adds it back to the output, making learning smoother. ResNet comes in various depths, such as ResNet-18, ResNet-50, and ResNet-101, with deeper versions using bottleneck blocks to reduce computational cost. It's widely used in tasks like image classification and object detection due to its high accuracy and scalability.

<br>

In this **ResNet** Implementation, we will be using the same dataset we used for **VGGNet**, (see above 👆). Let us observe the steps taken to carry out the implementation of ResNet 🔽

 - Implementing residual blocks of ResNet, including convolutional layers, batch normalization, ReLU activation, and residual connections - using **nn.Conv2d, nn.BatchNorm2d, nn.Sequential** and **nn.Identity**
 - Designing a **ResNet-18** model ▶️ Configuration with 18-layer. You can view the **[Original ResNet Paper](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1512.03385)** for having a reference. Below is a table, which includes information about different **ResNet** architecture 🔻<br><br>
![enter image description here](https://neurohive.io/wp-content/uploads/2019/01/resnet-architectures-34-101.png)
> NOTE : Due to computation resource limitations, we will stick to the architecture of **ResNet-18**.

 -  Training the model on the dataset that was used in the implementation of **VGGNet** Architecture (see above 👆).
 - Applying techniques to prevent overfitting to improve the results.
 - Saving the weights of the trained neural network that provides the best results.
 - Discussing the results and providing relevant graphs🔻
	 - Reporting training accuracy, training loss, validation accuracy, validation loss, testing accuracy, and testing loss.
	 - Plotting the training and validation accuracy over time (epochs).
	 - Plotting the training and validation loss over time (epochs).
	 - Generating a confusion matrix using the model's predictions on the test set.
	 - Calculating and reporting other evaluation metrics such as Precision, Recall and F1-score.

<br>
<br>
<br>

### [Time-Series Forecasting using RNN](https://github.com/SoubhikSinha/Grad-Deep-Learning/blob/main/RNN%20-%20Time%20Series%20Forecasting.ipynb) 👇<br>
Let us first understand what is meant by "*Time Series*"🔻
> A **time series** is a sequence of data points collected over time, where the order of the data matters. It’s used to track changes and identify patterns, such as trends or recurring cycles, across various fields like finance, weather, and healthcare. For example, daily stock prices, hourly electricity usage, or a week of heart rate data from a fitness tracker are all time series. What makes time series unique is its temporal nature—each data point is connected to the next by the passage of time, which allows us to analyze trends (e.g., an upward or downward direction), seasonality (e.g., regular patterns like holiday sales spikes), or unexpected changes (like a sudden drop in sales). By studying time series, we can forecast future values, detect anomalies, and better understand patterns in the data. Whether through simple statistical tools like moving averages or advanced methods like neural networks, time series analysis is a powerful way to make sense of time-based data.

<br>

For this notebook, we will consider the implementation of RNN (Recurrent Neural Network) to practicaly understand the features of a Time-Series Data and its analysis. But first, we shall also need to understand what exactly RNN is🔻
> A **Recurrent Neural Network (RNN)** is a type of neural network designed to handle sequential data by remembering information from earlier inputs to influence its understanding of later ones. This makes it ideal for tasks where context matters, such as language translation, speech recognition, or time series forecasting. Unlike regular neural networks, which process each input independently, RNNs maintain a "memory" by passing information from one step to the next through a hidden state. At each step, the network takes the current input and its previous state to calculate a new state and, if needed, an output. While this makes RNNs powerful for sequences, they can struggle with remembering long-term dependencies due to issues like vanishing or exploding gradients, where learning either stalls or becomes unstable. To address this, improved versions like Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) were developed, allowing the network to "decide" what information to keep or forget. These advancements have made RNNs a go-to for applications like predicting stock prices, translating languages, or even generating music.

<br>

Let us look at the steps, that were taken / should be taken when working with Time-Series Data and RNN 🔽

 - Selecting a Time-Series Dataset🔻
	 - Reading, Preprocessing and Printing the main statistcis about the dataset
	 - Data Visualization
	 - Preparing the dataset for training, e.g. ***normalizing, converting features to categorical values and splitting it into training, testing and validation sets***
 - Building RNN🔻
	 - Building the RNN model architecture
	 - Training RNN model and trying various setups and hyperparameters tuning to obtain an expected accuracy
 - Saving the weights of the trained neural network hat provides the best results
 - Discussing / Reporting the results and providing relevant visualization results🔻
	 - Reporing training accuracy, training loss, validation accuracy, validation loss, testing accuracy, and testing loss
	 - Plotting the training and validation accuracy over time (epochs)
	 - Plotting the training and validation loss over time (epochs)

<br>
<br>
<br>

### [Sentiment analysis using LSTM](https://github.com/SoubhikSinha/Grad-Deep-Learning/blob/main/LSTM%20-%20Sentiment%20Analysis.ipynb) 👇<br>

Let us first talk about the domain of the problem🔻
> **Sentiment analysis** is a way of understanding the emotions or opinions expressed in text, often used to classify it as positive, negative, or neutral. It's a key part of natural language processing and helps businesses and researchers make sense of large amounts of text, like product reviews, tweets, or customer feedback. For example, companies use sentiment analysis to figure out if people are happy with their products or to gauge reactions to a new campaign. It works by cleaning and analyzing text, breaking it down into patterns, and using machine learning or deep learning models to predict the emotional tone. Whether it’s monitoring social media, moderating harmful content, or doing market research, sentiment analysis turns words into valuable insights that help make informed decisions.

<br>

Now, let us understand what is LSTM🔻
> **Long Short-Term Memory (LSTM)** is a type of neural network designed to handle and learn from sequences of data, like text, time series, or audio, by remembering information over long periods. It’s an improvement over traditional Recurrent Neural Networks (RNNs), which often struggle to retain information as sequences get longer. LSTMs solve this with a clever design that includes a "forget gate," allowing them to decide what information to keep or discard as they process data. This makes them ideal for tasks like predicting the next word in a sentence, forecasting stock prices, or translating languages. Unlike basic RNNs, which can forget or get overwhelmed by long-term dependencies, LSTMs can focus on the right context, whether it’s remembering the beginning of a story or patterns in a lengthy time series. Their ability to capture meaningful patterns in sequences has made them a key tool in areas like natural language processing, speech recognition, and anomaly detection.

<br>

Let us understand how we can implement LSTM for the above-mentioned problem 🔽

 - Selecting a Sentiment Analysis based Dataset🔻
	 - Reading, Preprocessing and Printing the main statistcis about the dataset
	 - Data Visualization
	 - Preparing the dataset for training, e.g. ***normalizing, converting features to categorical values and splitting it into training, testing and validation sets***

-  Building LSTM🔻
	- Building the LSTM model architecture
	- Training LSTM model and trying various setups and hyperparameters tuning 
	- Saving the weights of the trained model, that returns best results
	- Reporting accuracy and loss for your network
	- Visualizing and analyzing the results

<br>

Now that we have experienced how to create an LSTM model, it's time to have a look over variations of LSTM architectures 🔽
> 🌟 A ***Gated Recurrent Unit (GRU)*** is a type of recurrent neural network (RNN) architecture designed to capture long-term dependencies in sequences, similar to Long Short-Term Memory (LSTM) units, but with a simplified structure. Both GRUs and LSTMs address the issue of vanishing gradients in long sequences, but GRUs combine the forget and input gates of LSTMs into a single update gate, making them more computationally efficient. The GRU uses two main components: the update gate, which decides how much of the past information to retain, and the reset gate, which determines how much of the previous memory to forget when updating the state. This streamlined design allows GRUs to perform similarly to LSTMs while requiring fewer parameters, making them ideal for tasks like time series analysis, speech recognition, and natural language processing where capturing temporal patterns is important.

> 🌟 A ***Bi-Directional LSTM (BiLSTM)*** is an advanced version of the traditional LSTM that processes data in both forward and backward directions, allowing it to capture context from both the past and the future. While a standard LSTM only looks at past information to make predictions, BiLSTM uses two LSTM layers—one reading the sequence from left to right, and the other from right to left. The outputs from both directions are then combined, creating a more complete understanding of the sequence. This ability to consider both past and future context makes BiLSTMs especially powerful for tasks like speech recognition, machine translation, and text analysis, where understanding the full context is crucial for making accurate predictions.

> 🌟  A ***Stacked LSTM*** is a neural network architecture that involves multiple layers of LSTM units stacked on top of each other. Unlike a single-layer LSTM, which processes sequences through just one set of LSTM cells, a stacked LSTM creates a deeper network by having the output of one LSTM layer feed into the next. This allows the model to learn more complex patterns and abstract representations from the data. Each layer in a stacked LSTM captures different levels of features or patterns in the sequence, enabling the model to understand both short-term and long-term dependencies more effectively. Stacked LSTMs are particularly useful for tasks that involve complex sequences, such as speech recognition, time-series forecasting, and natural language processing, as they provide the model with a greater capacity to learn from intricate data structures.

<br>

If we think of creating an improved LSTM, below are the steps that can be followed for the same and a comparative study 🔽
- Building an improved LSTM🔻
	- Using any other version of LSTM model architecture to improve the results. *E.g. Gated Recurrent Unit (GRU), Bidirectional LSTM, Stacked LSTM*
	- Training the improved LSTM model and trying various setups and
hyperparameters tuning
	- Saving the weights of the trained model, that returns best results
	- Reporting and loss for your network
	- Visualizing and Analyzing the results

<br>
<br>
<br>

### [Autoencoders for Anomaly Detection](https://github.com/SoubhikSinha/Grad-Deep-Learning/blob/main/Auto-Encoders%20_%20Anomaly%20Detection.ipynb) 👇<br>

Let us understand what is *Anomaly Detection* 🔽
> ***Anomaly detection*** is the process of identifying patterns or observations in data that do not conform to expected behavior, often referred to as "outliers" or "anomalies." These anomalies can indicate rare events, errors, fraud, or other significant occurrences, depending on the context. In machine learning, anomaly detection algorithms are trained to recognize typical patterns within a dataset, and then flag data points that deviate significantly from these patterns. It is commonly used in various fields, such as fraud detection in financial transactions, identifying network intrusions in cybersecurity, detecting equipment failures in industrial settings, or finding abnormal behaviors in health monitoring. The key challenge in anomaly detection is determining what constitutes "normal" behavior, which can vary depending on the application and the type of data being analyzed.

<br>

Now let us discuss about the technique to be used : Auto-Encoders 🔽
> ***Autoencoders*** are a type of neural network used for unsupervised learning, typically for tasks like dimensionality reduction, feature learning, and anomaly detection. They consist of two main components: an encoder and a decoder. The encoder compresses the input data into a lower-dimensional representation, called the latent space or bottleneck, while the decoder attempts to reconstruct the original input from this compressed form. The model is trained to minimize the difference between the input and the reconstructed output, often using a loss function like mean squared error. Autoencoders are useful for learning efficient data representations, and they can be applied in tasks such as noise reduction, anomaly detection, and feature extraction. Variants like variational autoencoders (VAEs) add probabilistic elements, making them useful for generative tasks like image generation.

<br>

Let us have a look how we may implement Auto-Encoder for the above-mentioned problem 🔽
- Selecting a Dataset
- Data exploration and preprocessing🔻
	- Reading, preprocessing, and printing the main statistics about the dataset
	- Cleaning and preparing the data for modeling ***(e.g. handling missing values, normalization, feature engineering)***
	- Data visualization : Revealing insights about the data and potential anomalies ***(e.g., histograms, time-series plots, scatter plots)***
	- Preparing the dataset for training : Dividing the preprocessed data into training, validation, and testing sets.

- Auto-Encoder model building 🔻
	- Implementing a standard [Autoencoder](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2003.05991) or [Variational Autoencoder (VAE)](https://arxiv.org/abs/1606.05908) for anomaly detection
	- Experimenting with architectures :  Building and training different
autoencoder architectures for anomaly detection. Considerng experimenting
with 🔻
		- Different layer types (Dense, LSTM for time series, Conv1D for sequential data)
		- Number of hidden layers and units
		- Activation functions (ReLU, sigmoid)

- Model evaluation and training🔻
	- Training each autoencoder architecture using the training data and validation data for monitoring performance.
	- Choosing appropriate evaluation metrics ***(e.g., reconstruction error,
precision-recall)*** for anomaly classification
	- Comparing the performance of the different architectures on the validation set : Discussing which architecture performs best and why (considering factors like reconstruction error and anomaly detection accuracy)

- Saving the weights of the trained network that provides the best results
- Discussing the results and providing relevant graphs 🔻
	- Reporting training accuracy, training loss, validation accuracy, validation loss, testing accuracy, and testing loss
	- Plotting the training and validation accuracy over time (epochs).
	- Plotting the training and validation loss over time (epochs).
	- Generating a confusion matrix using the model's predictions on the test set.
	- Calculating and report other evaluation metrics such as Precision, Recall and F1-Score

<br>
<br>
<br>

### [Build Transformer with PyTorch](https://github.com/SoubhikSinha/Grad-Deep-Learning/blob/main/Transformer%20-%20PyTorch.ipynb) 👇<br>
Let us discuss about ***"Transformer"*** 🔽
> A ***Transformer*** is a deep learning model architecture that has revolutionized natural language processing (NLP) and other sequential data tasks. Unlike traditional recurrent neural networks (RNNs) and LSTMs, which process data step-by-step, the Transformer uses a mechanism called **self-attention** to process all parts of the input simultaneously. This allows the model to capture relationships between words or tokens regardless of their position in a sequence, making it highly efficient and scalable for tasks like language translation, text generation, and image processing.<br>
The core components of the Transformer are the **encoder** and **decoder**, each consisting of multiple layers. The encoder processes the input data and generates representations, while the decoder uses these representations to produce the output. Self-attention allows each word or token in a sequence to attend to (or focus on) other words in the sequence, helping the model understand context in a flexible and dynamic way. Transformers have become the foundation for many state-of-the-art models, including ***GPT, BERT, and T5***, due to their ability to handle long-range dependencies and process data in parallel, leading to faster training and better performance on complex tasks.

<br>

Let us dive into the implementation steps of Transformer 🔽

 1. Data Exploration and Preprocessing ⬇️
	 - Select a Dataset
	 - Data Exploration🔻
		- Reading, preprocessing, and printing the main statistics about the dataset
		- Data visualization : Revealing insights about the data and potential anomalies ***(e.g., polarity distribution, word count distribution, vocabulary size etc.)***

	- Text Preprocessing 🔻
		- Text Cleaning ▶️ Removing punctuation, stop words, and unnecessary characters
		- Text Lowercasing ▶️ Ensuring all text is lowercase for consistent representation
		- Tokenization ▶️ Breaking down the text into individual words (or tokens)
		- Vocabulary Building ▶️ Creating a vocabulary containing all unique tokens encountered in the dataset
		- Numerical Representation ▶️ Converting tokens into numerical representations using techniques like word embedding ***(e.g., Word2Vec, GloVe)***

<br>

2. Model Construction ⬇️
	- Embeddings and Positional Encoding :  Defining an embedding layer to map tokens into numerical vectors. ***(NOTE : If using pre-trained embeddings, ensure they are compatible with your model's input dimension)***
	- Implement the core Transformer architecture🔻
		- Encoder ▶️ Utilizing **nn.TransformerEncoder** with multiple **nn.TransformerEncoderLayer** instances. Each layer typically comprises a multi-head self-attention mechanism, a feed-forward layer, and layer normalization.
		- Decoder ▶️ Employing **nn.TransformerDecoder** with multiple **nn.TransformerDecoderLayer** instances. These layers incorporate masked self-attention, multi-head attention over the encoder outputs, and a feed-forward layer with layer normalization.

- Depending on your task **(e.g., classification, sequence generation)**, defining an appropriate output layer **(NOTE : For classification tasks, you might use a linear layer with a softmax activation function)**

<br>

3. Training the Transformer ⬇️
	- Preparing for Training🔻
		- Dividing the preprocessed data into training, validation, and testing sets using a common split ratio (e.g., 70:15:15 or 80:10:10)
		- Choosing an appropriate loss function ***(e.g., cross-entropy loss for classification)*** and an optimizer ***(e.g., Adam)*** to update model parameters during training
	
	- Training Loop 🔻
		- Forward Pass ▶️ Passing the input data through the Transformer model to generate predictions
		- Calculating Loss ▶️ Computing the loss between predictions and true labels using the chosen loss function
		- Backward Pass ▶️ Backpropagating the loss to calculate gradients for each model parameter
		- Updating Parameters ▶️ Utilizing the optimizer to update model parameters based on the calculated gradients

<br>

4. Evaluation and Optimization ⬇️
	- After each training epoch, assessing the model's performance on the validation set ▶️ Monitoring metrics like accuracy or loss to track progress.
	- Exploring various optimization techniques to improve the performance of your Transformer model 🔻
		- Regularization (L1/L2) ▶️ Penalizing large model weights to prevent overfitting
		- Dropout ▶️ Randomly dropping out neurons during training to introduce noise and reduce overfitting
		- Early stopping ▶️ Halting training when validation performance plateaus to prevent overtraining
		- Learning Rate Tuning ▶️ Experiment with different learning rates to find the optimal value for convergence and performance

	- Saving the weights of the model that provides the best results.
	- Discussing the results and providing the following graphs 🔻
		- Reporting training accuracy, training loss, validation accuracy, validation loss, testing accuracy, and testing loss
		- Plotting the training and validation accuracy over time (epochs)
		- Plotting the training and validation loss over time (epochs)
		- Generating a confusion matrix using the model's predictions on the test set
		- Calculating and reporting other evaluation metrics such as ***Precision, Recall and F1-Score***
		- Plotting the ROC curve
