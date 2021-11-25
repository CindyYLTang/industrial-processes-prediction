# industrial-processes-prediction
## INTRODUCTION
Our client is a leading Hong Kong-based textile and apparel manufacturer, offering innovative and sustainable solutions to create premium cotton shirts for global brands. The client has developed and extended its business rapidly on a global scale by progressively constructing its vertically integrated supply chains or operating its ‘Integral’ project and in this condition, and the productivity and the efficiency have already improved significantly because of the innovative development for the past few years. The client is a market-driven company that progressively and dynamically innovates itself, aspiring to make a revolutionary difference in the traditional textile and apparel industry so that it can keep the leading position in the industry as well. Therefore, one of the ways for the client to further increase its productivity and efficiency is to have better control in the production process of products, thus in this project, we mainly focus on the predictions of actual width and shrinkage rate, two of the most important industrial process, based on the historical data given by the client.

Having a statistical analysis on the process of production, the client can have a systematical understanding of the production process so that the client would know what are the key factors that influence the production process, and thus, have better quantitative control on them. By doing so, the client can maximize its utility while reducing the cost of production so that it can have a much better financial performance for future perspectives. With the given dataset of actual width and shrinkage rate, we have applied random forest model, neural network model on the data that are separated into the training dataset and the test dataset.

## PROBLEM  THE COMPANY IS FACING - PAIN POINTS
To produce any cotton shirts, the material dosage should be minimized. However, due to various factors, the width of fabric rolls is uncertain. Hence the minimum required length of fabric is difficult to determine and it involves the interaction of a large number of variables, especially in a categorical way that needs to be optimized in the modeling process. At the same time, fabric shrinkage rates are also hard to predict in the same sense involving large numbers of factors. For example, theoretically, the fabric shrinkage ratio is affected by the following variables but to an unknown extent: 1. The type, count and density of the warp yarns. 2. The type, count and density of the weft yarns. 3. Weave structure (Kadi and Karnoub,2015: The Effect of Warp and Weft Variables on Fabric’s Shrinkage Ratio). The client's current model is a simple model with excel formulas and cannot incorporate complicated relationships and cannot process large volumes of data, especially categorical data. Hence, the client needs to discover a trained model to reliably predict the fabric dimensional change with bulk raw data as input.
## CODE
1. shrinkage_and_actual_width_eda.ipynb under folder eda is the exploratory data analytics for actual witdh and shrinkage dataset
2. nn_predict.ipynb under folder actual_width_prediction is the prediction of action width with neural network model
3. rf_predict.ipynb under folder actual_width_prediction is the prediction of action width with random forest model
4. nn_predict.ipynb under folder shrinkage_prediction is the prediction of shrinkage with neural network model
5. rf_predict.ipynb under folder shrinkage_prediction is the prediction of shrinkage with random forest model
## MODEL
### Prediction of Actual Width
#### The data used
The data we used for actual width prediction is internal data provided by the client. The client would like to provide estimates of actual width based on the rest features to facilitate the decision-making process and enhance production efficiency. Four Excel files containing historical data from April 2020 to April 2021 were provided by the company as our raw data.

After concatenation, there are in total 308249 rows and 25 columns of actual width related records. Among them, actual width is our dependent variable, while the rest of the 24 columns of data would be the dependent variables describing the potential determinants of the actual width statistics. As instructed by the client, the actual width figure will fluctuate against design width, and design width will be considered as the major contributor to the actual width statistics. The rest of the variables, for instance, reed_no will influence how much the actual width statistics deviate from the design width statistics.

During the exploratory data analytics, we analyzed data sets to summarize their main characteristics, with statistical graphics data visualization methods. The first thing we noted is that there are two types of variables, i.e, numeric variables and categorical variables. Numerical data are expressed in numbers and are continuous. Categorical data is, however, discrete, and is a type of data that is stored into groups or categories with the aid of names or labels. As for categorical data, there are two subtypes of data, i.e, nominal data where data sets are names or labels. Ordinal data includes elements that are ranked, ordered. Our categorical data all fall into the type of nominal data as they are not ranked and ordered. We distinguish data with Python command table.info(), where float number stands for numeric values and object stands for categorical variables. We then exclude those variables that don’t have a business meaning, and are just a way to distinguish different rows of data, which hence are unimportant or have lower correlation with the target variables including 'ETD-出货日期', 'GF_NO-品名',' job_no-排单号', 'fabric_no-布号', 'fnsend_no-后整送布单', 'Quantity-卷长’.

We utilized different methods in evaluating numeric and categorical variables. For numeric variables, we utilized the correlation method and scatterplot. For categorical variables, we utilized box-plot to display the distribution of data based on quantile, minimal, maximal, and outliers.

Characteristics of numeric variables are shown below. For the dependent variable actual width, we noted that it is a weighted average of several normal distributions. For independent variables, we noted that design width represented a strong positive correlation with actual width, with Pearson correlation = 0.97. Other variables are randomly distributed, and no pattern was noted for the rest of the numeric variables.

For categorical variables, no pattern was noted when they are plotted against actual width individually. Further correlations and relationships can only be dug when they are explained collaboratively.

#### Data Pre-processing
We pre-processed the raw data for the model training purpose. Firstly, we impute all missing values with value 0, since as discussed with the client, the missing data represent null values.

Secondly, in order to make categorical variables understandable for computers, we apply one hot encoding method with Python command pandas.get_dummies that derives the categories based on the unique values in each feature. For instance, to represent Fabric Type in a binary sense, we derived 8 columns for each fabric type and the value will be assigned to 1 if the row belongs to a certain fabric type and 0 if it does not belong to a certain fabric type. Before implementing the one-hot encoding function, the dataset matrix is( 308249,18)  with 8 numeric and 10 categorical variables. After implementing the function, the data set matrix is (308249, 631) without numerical columns and (308249, 639) with numerical columns.

#### Model Training and Selection
To predict actual width, we randomly divided 80% and 20% of raw data into the training set and test set with seed 1234. As mentioned in part 3b, we trained four different machine learning algorithms (Linear regression, Neural Network, Random Forest) with the same training dataset. Also, since actual width and design width have an extremely high correlation, we noted that the one with the dependent variable being actual width generates unsatisfactory results. Hence, we trained two sets of models, one with target variables being actual width, another with target variable being the discrepancy between actual width and design width, i.e, actual width - design width. We will further compare the variance between the two and determine which model can make better predictions. To compare the prediction accuracy of the trained models in the test set, we used mean-square error as our measurement. 

#### Neural Network architecture
For Neural Network, we chose the following parameters:
i. Network architecture, we chose fully connected NN over Convolutional as we are facing a black box situation and make no assumptions about the features in the data.
ii. Loss function: we chose MSE over Cross_entropy for classification as we have a continuous dependent variable
iii. Network input matrix,  we determine that (m,n)=(308249,639). When there are only 1 hidden layer, ℎ=x∙w_ℎidden+b_ℎidden and y_pred=ℎ∙w_output. When there are more hidden layers: ℎ_1=x∙w_ℎidden1+b_ℎidden1, ℎ_2=ℎ_1∙w_ℎidden2+b_ℎidden2, and y_pred=ℎ_2∙w_output
vi: Parameter Tuning: we used the following parameters in the tuning process:
Learning Rate: [0.002, 0.003, 0.004]
Width: no of neurons, [5,10,20], we originally chose [50,200,2000], but found out the result to be unsatisfactory.
Depth: no of layers, range(5,9)
Batch_size = 500
Steps=100000
1 epoch=308249/500 =616, 162 epochs in total
After the tuning process, we noted that MSE, mean and standard deviation converges to a certain value, which means the number of steps is enough and our training is complete.

#### Random forest architecture
We utilized the following parameters in the tuning process.
Criterion: MSE, MAE
n_estimators, which stands for number of trees: linspace(start = 200, stop = 2000, num = 10)
max_features which stands for the max feature for each tree: ['auto', 'sqrt']
max_depth which stands for the max depth for each tree: linspace(10, 100, num = 10)
min_samples_split which stands for the minimum number of samples required to split an internal node: [2, 5, 10]           
min_samples_leaf The minimum number of samples required to be at a leaf node: [1, 2, 4]             
Bootstrap: [True, False]
We noted that the final parameter: bootstrap=False,max_features='sqrt', max_depth=20,min_samples_split=10, n_estimators=1200,min_samples_leaf=2. If we take one node to look deeper, we noted that the result of the leftmost node would be -1.71.

### Prediction of Shrinkage
#### The data used
The data we used for shrinkage prediction is internal data provided by Esquel group. The company would like to provide estimates of shrinkage in warp and weft direction based on the rest features to facilitate decision-making process and enhance production efficiency.  Historical data are provided for our training purpose.
There are in total 113481 rows and 20 columns of shrinkage related records provided by the client. Among them, warp shrinkage and weft shrinkage are our dependent variables, while the rest of the 18 columns of data would be dependent variables describing the potential determinants of the shrinkage statistics. Two models are going to be constructed in order to predict warp shrinkage and weft shrinkage accordingly.
Similar to actual width prediction, we analyzed data sets to summarize their main characteristics, with statistical graphics data visualization methods. The first thing we noted is that there are two types of variables, i.e, numeric variables and categorical variables.
We utilized different methods in evaluating numeric and categorical variables. For numeric variables, we utilized correlation method and scatterplot. For categorical variables, we utilized box-plot to display the distribution of data based on quantile, minimal, maximal and outliers.
Characteristics of  variables are shown below: We noted that both warp density and weft density are normally distributed. However, independent variables are randomly distributed against dependent variables, and have little correlation with dependent variables. No pattern was noted for the independent variables.
Hence, from our exploratory data analytics, we can note that we cannot simply apply regression methods to the datasets. Instead, machine learning methods like random forest and neural networks should be utilized in our case.
### Data Pre-processing
We pre-processed the raw data for the model training purpose. Firstly, we impute all missing values with value 0, since as discussed with the client, the missing data represent null values. Secondly, we corrected typos from the original dataset. For example, we changed the value 0.8. to 0.8. Thirdly, we applied one hot encoding method with Python command pandas.get_dummies as is in the actual with part. Before implementing the function: we had a matrix of (113481,12)  with 4 numeric and 8 categorical variables. After implementing the function, the data set matrix is  (113481, 1175) with numerical columns

### Model Training and Selection
To predict shrinkage, we randomly divided 80% and 20% of raw data to the training set and test set with seed 1234. As it mentioned in part 3b, we trained four different machine learning algorithms (Linear regression, Neural Network, Random Forest) with the same training dataset. Also, we noted that the dataset has extremely high cardinality,  which will obscure the order of feature importance and make the accuracy lower, we also used alternatives like distributed random forest. Other methods can include a single feature with n numeric values and log n Boolean features representing n values. After that, to further deduct cardinality, we found out from the first model and also after discussion with client as for the business sense importance that warp yarn count and weft yarn count represents lower significance, and we also tried sets of models without the two variables.

### Neural Network architecture
We used the same sets of parameters as in actual width prediction thought the below parameter tuning process and we will not repeat it here.After the tuning process, we noted that MSE, mean and standard deviation converges to a certain value, which means number of steps is enough and our training is complete.

### Random forest architecture
We utilized the same parameters in the tuning process as in actual width prediction. We noted that the final parameter: bootstrap=False,max_features='sqrt', max_depth=20,min_samples_split=10, n_estimators=1200,min_samples_leaf=2.
