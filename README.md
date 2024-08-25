# deep-learning_challengev2

## Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization.

## Preprocess the Data
- Create a dataframe containing the charity_data.csv data , and identify the target and feature variables in the dataset
- Drop the EIN and NAME columns 
- Determine the number of unique values in each column 
- For columns with more than 10 unique values, determine the number of data points for each unique value
- Create a new value called Other that contains rare categorical variables
- Create a feature array, X, and a target array, y by using the preprocessed data 

## Split the preprocessed data into training and testing datasets 

## Scale the data by using a StandardScaler that has been fitted to the training data

## Compile, Train and Evaluate the Model 
- Create a neural network model with a defined number of input features and nodes for each layer
- Create hidden layers and an output layer with appropriate activation functions
- Check the structure of the model
- Compile and train the model
- Evaluate the model using the test data to determine the loss and accuracy
- Export your results to an HDF5 file named AlphabetSoupCharity.h5

## Optimize the Model
- Repeat the preprocessing steps in a new Jupyter notebook
- Create a new neural network model, implementing at least 3 model optimization methods
- Save and export your results to an HDF5 file named AlphabetSoupCharity_Optimization.h5

## Write a Report on the Neural Network Model

### Overview of the analysis: 
The purpose of the analysis is to create model that will determine if an AlphabetSoup Charity campaign will be successful or unsuccessful based on a 34000 x 12 set of data about past campaigns. 
Our best resulting model only performs at 72.67% which is useful, but not definitive.  

### Results: Using bulleted lists and images to support your answers, address the following questions:

### Data Preprocessing
- What variable(s) are the target(s) for your model?
  - IS_SUCCESSFUL is our X target
- What variable(s) are the features for your model?
  - Our y features are
    - APPLICATION_TYPE
    - AFFILIATION
    - CLASSIFICATION
    - USE_CASE
    - ORGANIZATION
    - STATUS
    - INCOME_AMT
    - SPECIAL_CONSIDERATIONS
    - ASK_AMT
- What variable(s) should be removed from the input data because they are neither targets nor features?
  - EIN
  - Name

### Compiling, Training, and Evaluating the Model

#### Model 1 : 
- Layers: 3x layers
  - to keep the complexity of the model low enough to be processed without undue time 
- Neurons: 193 neurons (128, 64, and 1 in output)
  - 128 is standard starting point in modeling moderately sized datasets.
  - subsequent layers are smaller, assuming that the data has been abstracted.
  - The final layer is only one neuron because it is creating a single classification-- yes/no, successful/not successful. 
- Activation : ReLU used in layer 1 and 2, Sigmoid used in the output.
  - ReLU because it is general purpose.
  - Sigmoid because this is a binary classification so Sigmoid allows us to reduce the data to binary.  

#### Model 2 :
Because Model 1 appeared to be overfitting, we added options to correct to a more underfit performance. 
- Layers: 2x layers
- Neurons: 129 neurons (128, 1 in output)
- Activation: ReLU for hidden and Sigmoid for output
- Regularization: L2, Dropout

#### Model 3 : 
Instead of manually fitting the neural complexities of the model, we chose to use the keras tuner to help us determine the best fit hyperparameters. 
- Layers: 4x layers (1x is a drop-out layer)
- Neurons: 449 neurons (416, 32, 1 in output)
- Activation: ReLU for hidden and Sigmoid for output
- Regularization: Dropout

### Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation
Of the models, the keras-tuned model is the most accurate with a 72.67% accuracy rate, however this is not much better than 1 at 72.3 or 2 at 71.72.  

For future work, I would like to see an explanation of data column types.  For example, I think "ORGANIZATION" may have no bearing on success. Some domain knowledge in this subject might be helpful to determine what does and does not affect the data.  It might be worth creating a model that would help determine causal columns in some way. Preprocessing differently could provide different results.  

Beyond updating the preprocessing to get a cleaner initial model setup, and because this is a relatively small dataset, our concern has to be overfitting.   When we are concerned with overfit data there are a few models to consider: 
-Random Forest - robust to overfitting, don't require feature scaling, and have more simplified pathways, good for non-linearity
-Gradient Boosting Machines (GBM) - learns on mistakes of prior machine learning
-Support Vector Machines (SVM) - robust to overfitting, useful if there were more features
-Logistic Regression - simple, fast, and if we were to use it first, other, deeper, models might not be needed 
-Pretrained Models- why do the work if someone else already has? 

# Resources
- functions from class, Kourt Bailey (TA) suggestions and help
- ChatGPT 4.0 (The Intern) 
- EDX Bootcamps 
- Data from IRS. Tax Exempt Organization Search Bulk Data Downloads. https://www.irs.gov/Links to an external site.
