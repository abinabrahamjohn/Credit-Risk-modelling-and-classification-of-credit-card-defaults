%Grid search for Random Forest Hyper Parameter Tuning. This script takes
%few minutes to complete

%Clear workspace, command window and close figures
clear all;
clc;
close all;

%Dataloading
%Selecting specific rows onwards as the original data set has 2 header rows and a
%redundant id column in the start
credit_default_data=readtable('default_of_credit_card_clients.xls','Range','B2:Y30002');

%Data Pre-processing : Transforming categorical variables of double type into categorical type
catColumns = {'EDUCATION', 'SEX', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','defaultPaymentNextMonth'};
catColumnsFilter = ismember(credit_default_data.Properties.VariableNames, catColumns);
for i = 1:length(catColumns)
    col = catColumns{i};
    credit_default_data.(col) = categorical(credit_default_data{:, col});
end


%Data cleaning
%Education has no value of '0','5','6' based on the data definiton, hence
%assigning it to one of the defined values '4' for others.
credit_default_data.EDUCATION(strcmpi(credit_default_data.EDUCATION,'5')) = {'4'};
credit_default_data.EDUCATION(strcmpi(credit_default_data.EDUCATION,'6')) = {'4'};
credit_default_data.EDUCATION(strcmpi(credit_default_data.EDUCATION,'0')) = {'4'};

%Marriage has no value '0' based on the data definition
credit_default_data.MARRIAGE(strcmpi(credit_default_data.MARRIAGE,'0')) = {'3'};

%remove correlated columns

%Feature scaling and normalization of numeric features
normalize(credit_default_data,'DataVariables','LIMIT_BAL');
normalize(credit_default_data,'DataVariables','AGE');
normalize(credit_default_data,'DataVariables','BILL_AMT1');
normalize(credit_default_data,'DataVariables','BILL_AMT2');
normalize(credit_default_data,'DataVariables','BILL_AMT3');
normalize(credit_default_data,'DataVariables','BILL_AMT4');
normalize(credit_default_data,'DataVariables','BILL_AMT5');
normalize(credit_default_data,'DataVariables','BILL_AMT6');
normalize(credit_default_data,'DataVariables','PAY_AMT1');
normalize(credit_default_data,'DataVariables','PAY_AMT2');
normalize(credit_default_data,'DataVariables','PAY_AMT3');
normalize(credit_default_data,'DataVariables','PAY_AMT4');
normalize(credit_default_data,'DataVariables','PAY_AMT5');
normalize(credit_default_data,'DataVariables','PAY_AMT6');

%Splitting data  into training and test data using holdout
%train: 80%, test: 20%
rng(110)
cv = cvpartition(size(credit_default_data,1),'HoldOut',0.2);
idx = cv.test;
% Separate to training and test data
train_credit_default_data = credit_default_data(~idx,:);
test_credit_default_data  = credit_default_data(idx,:);

%Downsampling of training data
%Other options are SMOTE and ADASYN- future work
training_counts=histcounts(train_credit_default_data.defaultPaymentNextMonth);
fprintf('The training dataset has %d 0s and %d 1s in the target.\n', training_counts(1), training_counts(2))
default_1_rows=find(train_credit_default_data.defaultPaymentNextMonth=='1');
default_0_rows=find(train_credit_default_data.defaultPaymentNextMonth=='0');
undersampled=default_0_rows(1:length(default_1_rows));
rowsToExtract=sort([undersampled;default_1_rows]);
train_credit_default_data=train_credit_default_data(rowsToExtract,:);
down_sampled_counts=histcounts(train_credit_default_data.defaultPaymentNextMonth);
fprintf('The downsampled training dataset has %d 0s and %d 1s in the target.\n', down_sampled_counts(1), down_sampled_counts(2))

%Feature Selection


%Data split into features and labels
train_credit_default_labels=train_credit_default_data(:,24);
train_credit_default_features=train_credit_default_data(:,1:23);


test_credit_default_labels=test_credit_default_data(:,24);
test_credit_default_features=test_credit_default_data(:,1:23);

%Initialising array to store grid search paramters and results
Metric=[];
%Predetermined range of trees, minimum leaves and 
%minimum number of predictors
num_trees=[1 20 40 60 80 100];
num_leaves=[1 10 20 30 40 50];
num_predictors=[5 10 15 20 23];
iteration=0;
for tree=num_trees
   for minleaf=num_leaves
       for numpred=num_predictors
           iteration= iteration+1
   treeRandomForest = TreeBagger(tree,train_credit_default_features,train_credit_default_labels,'ClassNames',{'0'  '1'},'Prior',[0.78 0.22],'MinLeafSize',minleaf,'Method','classification','NumPredictorsToSample',numpred);
   treeRandomForestResult=predict(treeRandomForest,test_credit_default_features);
[predRF, PosteriorRF]=predict(treeRandomForest,test_credit_default_features);
confusion_matrix_RF=confusionmat(test_credit_default_labels.defaultPaymentNextMonth,categorical(treeRandomForestResult));

%Calculating perfromance metrics corresponding to grid search parameters)
[accuracyRF,precisionRF, recallRF, specificityRF,fscoreRF] = PerformanceMetrics(confusion_matrix_RF);

%Array to store grid search paramters and results
Metric=[Metric;tree minleaf numpred accuracyRF precisionRF recallRF specificityRF fscoreRF;]
       end
   end
end

%determining the maximum values for each row to determine the best
%performance metric
maxValues=max(Metric);
optimizedValues=[];
fprintf('Hyperparameters for each objective\n')
fprintf('   trees     minleaf   numpred   accuracyRF   precisionRF   recallRF  specificityRF  fscoreRF\n')
forbestAccuracy=Metric(find(Metric(:,4)==maxValues(4)),:)
fprintf('   trees     minleaf   numpred   accuracyRF   precisionRF   recallRF  specificityRF  fscoreRF\n')
forbestPrecision=Metric(find(Metric(:,5)==maxValues(5)),:)
fprintf('   trees     minleaf   numpred   accuracyRF   precisionRF   recallRF  specificityRF  fscoreRF\n')
forbestRecall=Metric(find(Metric(:,6)==maxValues(6)),:)
fprintf('   trees     minleaf   numpred   accuracyRF   precisionRF   recallRF  specificityRF  fscoreRF\n')
forbestSpecificity=Metric(find(Metric(:,7)==maxValues(7)),:)
fprintf('   trees     minleaf   numpred   accuracyRF   precisionRF   recallRF  specificityRF  fscoreRF\n')
forbestF1score=Metric(find(Metric(:,8)==maxValues(8)),:)
