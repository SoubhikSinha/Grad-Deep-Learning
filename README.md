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
<br>

### **[Data Analysis, ML Models and PyTorch](https://github.com/sricks404/Grad-Deep-Learning/blob/main/Data%20Analysis.%20ML%20Models%20and%20PyTorch.ipynb)** 👇<br>

This notebook acts as a referesher to your Machine Learning concepts. Later in this notebook, you will find a practical use case of **[PyTorch](https://pytorch.org/)** - which is a popular Deep Learning framework, used widely among researchers, academic institutions, students and leading companies in the tech industry. 
<br>
<br>
This notebook is divided into three steps, each containing unique tasks (sub-steps). We'll go through these sub-steps one by one to help you quickly refresh your Data Analysis and Machine Learning concepts. However, I strongly recommend that you go through the entire notebook and skim through the code cells as well🔽
<br>
> **NOTE:** If you're completely new to Deep Learning, no worries! I've created a separate repository, **[Grad-Machine-Learning](https://github.com/sricks404/Grad-Machine-Learning)**, which can help you get started with Machine Learning to have a deep practical understanding of the subject. Once you've gone through that, you'll feel confident enough to dive into the world of Deep Learning.

<br>
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
