### Project Title

Creation of Machine Learning models that can be used to predict weight loss using exercise and dietary information for use by stakeholders that want to understand the features that are key for weight loss.

**Author**

Yemi Falokun

#### Executive summary

Machine learning (ML) is a branch of artificial intelligence (AI) that enables computers to “self-learn” from training data and improve over time, without being explicitly programmed. This can be used to predict weight loss, as people who lose weight tend to have certain patterns in their behavior. For example, they may eat less, exercise more, or change their diet. By analyzing data from people who have lost weight, machine learning algorithms can identify these patterns and use them to predict whether or not someone will be successful in their weight loss efforts.

This technology has the potential to revolutionize the way we think about weight loss. Instead of relying on traditional methods like dieting and exercise, we may be able to use machine learning to create personalized weight loss plans that are more likely to be successful.

#### Rationale
Weight loss is a difficult and often frustrating process, but it can be life-changing. Here are just a few reasons why anyone should care about weight loss:

- Weight loss can improve your health, extra weight puts a strain on your heart, lungs, and joint and losing this weight can help to reduce your risk of developing health problems like heart disease, stroke, and type 2 diabetes
- Weight loss can improve your self-esteem, the extra weight can make people feel self-conscious and uncomfortable in your own skin. Losing weight can help you to feel more confident and comfortable in your own body
- Weight loss can improve your life expectancy. Carrying around extra weight can shorten your lifespan. Losing weight can help you to live a longer, healthier life

#### Research Question
There are many different research questions that could be asked about weight loss. Some examples include:
- What are the exercises that has the most impact on weight loss
- Does diet have a positive impact on weight loss
- With the dataset available, does Age, weight and gender have an impact on weight loss 


#### Data Sources
The following datasets will be used for this project:

- Fitbit weight data loss dataset from https://www.kaggle.com/datasets/arashnic/fitbit
- Weight data loss dataset from https://www.kaggle.com/datasets/tombenny/foodhabbits

#### Methodology
The following Machine Learning models will be used:
- Linear regression is a statistical method that can be used to understand the relationship between variables that contribute to weight loss, the variables would be exercise and weight loss
- Decision tree is another model that will be used to predict weight loss. It does this by creating a series of if-then statements that are based on the data. For example, if someone eats less or exercises more, then they are more likely to lose weight
- Compare results of Classifiers like K Nearest Neighbors (KNN), Logistic Regression, Support Vector  Machines (SVC) on the dataset to determine the model with greatest accuracy


#### Results

This project uses the ML skills acquired during this course to build ML models predicting Calories burnt or weight loss using data from exercise by Fitbit users and Dietary information provided by users.

As described above in [Data Sources](#data-sources), I examine a couple of datasources. The [foodDiet.csv](https://github.com/yemifalokun/predicting-weight-loss/blob/main/data/foodDiet.csv) which includes 78 records with 7 features was used to build a Linear Regression ML model in the [predicting-weight-loss-using-diet-data-Linear-Regression.ipynb](https://github.com/yemifalokun/predicting-weight-loss/blob/main/notebooks/predicting-weight-loss-using-diet-data-Linear-Regression.ipynb)

The training model accuracy from this modeling is 0.492845 which means that the model correctly classified 49.28% of the instances. For Testing model, the accuracy score of -0.774792 and a  Mean Absolute Error of 0.202576  which means that the model was able to correctly classify only 22.52% of the data.

| Metric       	| Values                   | 
|-------------------	|:---------------------------	|
| Training 𝑅2 Score     | 0.492845                      |  
| Training RMSE         | 0.206424                        |  
| Training MAE          | 0.148942                        |  
| Testing 𝑅2 Score      | -0.774793                         |  
| Testing RMSE          | 	0.266211                        |  
| Testing MAE           | 0.202576                         |  
|                       |                               |  

This may be due to the number of records and features of the dataset. Based on these results, the models listed in the [Methodology](#methodology) section will use the [dailyActivity_merged.csv](https://github.com/yemifalokun/predicting-weight-loss/blob/main/data/dailyActivity_merged.csv) dataset.

Results from the models are described below:

##### Notebook 1 



##### Notebook 2




##### Notebook 3


| Model Name        	| Train Time (s)                      | Train Accuracy                | Test Accuracy 	                | 
|-------------------	|:---------------------------	|:---------------------:	|:----------------------:	|
| Logistic Regression   | 0.322                         | 0.8872047448926502        | 0.8875940762320952                 |  
| KNN                   | 55.8                          | 0.8846033783080711        | 0.8807963097839281                  |  
| Decision Tree	        | 0.376                         | 0.8911935069890049        | 0.884761673545359                 |  
| SVM                   | 24.4                          | 0.8873087995560335        | 0.8875131504410455                 |  
|                       |                               |                           |                        	| 

Using Grid Search to create models with the different parameters and evaluate the performance metrics

| Model Name        	| Train Time (s)                      | Best Parameters                                          | Best Score 	                | 
|-------------------	|:---------------------------	|:-------------------------------------------------:	         |:----------------------:	|
| Logistic Regression   | 64                            | C:0.001, penalty:l2, solver: liblinear	                     | 0.8872394393842521                |  
| KNN                   | 302                           | n_neighbors: 17                                                | 0.8855397848500199                 |  
| Decision Tree         | 15.7                          | criterion: entropy, max_depth: 1, model__min_samples_leaf: 1   | 0.8872394393842521                  |  
| Logistic Regression   | 490                           | C: 0.1, kernel: rbf                                            | 0.8872394393842521                 |  
|                       |                               |                                                                |                        	| 



#### Outline of project
- [predicting-weight-loss-notebook1-linear-regression](https://github.com/yemifalokun/predicting-weight-loss/blob/main/notebooks/predicting-weight-loss-notebook1-linear-regression.ipynb)

- [predicting-weight-loss-notebook2-decision-tree](https://github.com/yemifalokun/predicting-weight-loss/blob/main/notebooks/predicting-weight-loss-notebook2-decision-tree.ipynb)

- [predicting-weight-loss-notebook3-comparing-classifiers](https://github.com/yemifalokun/predicting-weight-loss/blob/main/notebooks/predicting-weight-loss-notebook3-comparing-classifiers.ipynb)

##### Contact and Further Information