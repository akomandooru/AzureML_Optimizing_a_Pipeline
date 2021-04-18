# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, I built and optimized an Azure ML pipeline using the Python SDK and a provided Scikit-learn logistic regression model. I used hyperdrive to optimize hyperparameters of the logistic regression model. As a next step, I used Azure AutoML to find another optimal model using the same input dataset so that I can compare the results and conclude if the HyperDrive optimized logistic regression model outperformed AutoML model. 

A high level solution architecture diagram is shown below for the two approaches 

![Diagram](images/arch.png "Solution Architecture")

_Step 1_: Start by setting up a train script (train.py); we will create a dataset and evaluate it using scikit-learn logistic regression model type. Optimize the hyperparameters of this logistic regression model using HyperDrive so that we have a trained logistic regression model.

_Step 2_: Next, find an optimal model for the same dataset in Step 1 using automated machine learning (AutoML). 

_Step 3_: Compare the results of the two methods and write a research report in the form of a Readme described here in this document.

## Summary
**The [data](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv)  is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. The classification goal is to predict if the client will subscribe to a term deposit (variable y). [S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing.](https://archive.ics.uci.edu/ml/datasets/bank+marketing)**

**From an accuracy metric standpoint, both models returned an accuracy of 91.6% and so are equally performant if we look at just accuracy. When we take the design and coding required to implement the two options, automated machine learning implementation model is significantly easy and thorough as AutoML takes care of selecting an algorithm and tuning its hyperparameters. So, AutoML comes out on top when we combine accuracy and development effort together.**

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

### Pipeline architecture 
[See overview](#Overview)

### Data set information
The [data](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. 

#### Input variables:
**bank client data:**  
1 - age (numeric)  
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')  
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)  
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')  
5 - default: has credit in default? (categorical: 'no','yes','unknown')  
6 - housing: has housing loan? (categorical: 'no','yes','unknown')  
7 - loan: has personal loan? (categorical: 'no','yes','unknown')  
related with the last contact of the current campaign:  
8 - contact: contact communication type (categorical: 'cellular','telephone')  
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')  
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')  
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.  
**other attributes:**  
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)  
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)  
14 - previous: number of contacts performed before this campaign and for this client (numeric)  
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')  
**social and economic context attributes:**  
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)  
17 - cons.price.idx: consumer price index - monthly indicator (numeric)  
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)  
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)  
20 - nr.employed: number of employees - quarterly indicator (numeric)  

#### Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

### Hyperparameter tuning
Hyperparameter tuning, also called hyperparameter optimization, is the process of finding the configuration of hyperparameters that results in the best performance. The process is typically computationally expensive and manual.

The code below shows the use of random parameter sampler, bandit policy, SKLearn estimator, and a hyperdrive configuration for the hyperdrive run. 

**#Specify parameter sampler**  
ps = RandomParameterSampling({
      "--C": uniform(0.1, 0.9),
      "--max_iter": choice(10, 50, 100)
    }
)

**#Specify a Policy**  
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

if "training" not in os.listdir():
    os.mkdir("./training")

**#Create a SKLearn estimator for use with train.py**  
est = SKLearn(source_directory = "./",
            compute_target=compute_target,
            vm_size='STANDARD_D2_V2',
            entry_script="train.py")

**#Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.**  
hyperdrive_config = HyperDriveConfig(hyperparameter_sampling=ps, 
                                     primary_metric_name='accuracy',
                                     primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                     policy=policy,
                                     estimator=est,
                                     max_total_runs=16,
                                     max_concurrent_runs=4)
### Classification algorithm
The code below inside train.py shows the use of logistic regression for the classification algorithm when using our custom coded scikit-learn regression run. 

model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

Logistic regression, despite its name, is a classification algorithm rather than regression algorithm. Based on a given set of independent variables, it is used to estimate discrete value (0 or 1, yes/no, true/false). 

**What are the benefits of the parameter sampler you chose?**
I chose RandomParameterSampling as the sampler as it is faster, cheaper, and supports early termination of lower performance runs. 

**What are the benefits of the early stopping policy you chose?**
I chose banditpolicy as it can automatically end poorly performing runs with an early termination policy. Early termination improves computational efficiency.

## AutoML
I used the following configuration for this automated machine learning run

automl_config = AutoMLConfig(
    compute_target = aml_compute,
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric='accuracy',
    training_data=ds,
    label_column_name='y',
    n_cross_validations=2)

I kept the experiment timeout to be 30 min, set the task to a 'classification' task, and used 'accuracy' as the primary metric.

The model generated by AutoML for the best run is VotingEnsemble. 


## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
Both models (Scikit-learn logistic regression and autoML) reported an accuracy of 91.6%. Scikit-learn logistic regression model required a large amount of manual intervention during development; this included setting up a training script, picking an algorithm, and tuning its hyperparameters to produce a trained model. In contrast, an autoML trained model was created with minimal manual steps and most everything including picking an algorithm and hyperparameter tuning was automatic. 

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**  
Let's take a look at input dataset label distribution shown below  
  
![Diagram](images/class-dist.png "Class (label) distribution")  
  
This dataset used for training our model has just 11% labeled as "yes" for clients subscribed for a term deposit. Class data is highly imbalanced in this dataset. It is likely that both of our trained models will likely predict a high rate of false positives given only 11% of the date is labelled with a "yes" for training. This bank need is to find customers that will sign up ("yes") for a term deposit and so accuracy metric is not a good metric. Some improvements that will produce a higher quality trained models are  
  
1. Collect more data with an even distribution of labels  
2. Change the model's performance metric (use AUC, recall, F1 score in addition to accuracy)  
3. Try a different algorithm for our Scikit-learn run (Decision trees and penalized learning model work better with imbalanced datasets) 

[Reference: How to Deal with Imbalanced Data](https://towardsdatascience.com/how-to-deal-with-imbalanced-data-34ab7db9b100)

## Proof of cluster clean up
**Last step in the notebook has code to delete the cluster**
