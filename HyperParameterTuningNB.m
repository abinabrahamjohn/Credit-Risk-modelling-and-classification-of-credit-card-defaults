%Clear workspace, command window and close figures
clear all;
clc;
close all; 

%Dataloading
%Selecting specific rows onwards as the original data set has 2 header rows and a
%redundant id column in the start
credit_default_data=readtable('default_of_credit_card_clients.xls','Range','B2:Y30002');

%Categorical data conversion
% Data Pre-processing : Transforming categorical variables into the categorical type
catColumns = {'EDUCATION', 'SEX', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'defaultPaymentNextMonth'};
catColumnsFilter = ismember(credit_default_data.Properties.VariableNames, catColumns); % boolean mask for categorical attributes
for i = 1:length(catColumns)
    col = catColumns{i};
    credit_default_data.(col) = categorical(credit_default_data{:, col});
end

%data cleaning
credit_default_data.EDUCATION(strcmpi(credit_default_data.EDUCATION,'5')) = {'4'};
credit_default_data.EDUCATION(strcmpi(credit_default_data.EDUCATION,'6')) = {'4'};
credit_default_data.EDUCATION(strcmpi(credit_default_data.EDUCATION,'0')) = {'4'};
credit_default_data.MARRIAGE(strcmpi(credit_default_data.MARRIAGE,'0')) = {'3'};

%feature scaling and normalization
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

%Splitting data  into training and test data
%Should be randomly picked
%train: 80%, test: 20%
rng(110)
cv = cvpartition(size(credit_default_data,1),'HoldOut',0.2);
idx = cv.test;
% Separate to training and test data
train_credit_default_data = credit_default_data(~idx,:);
test_credit_default_data  = credit_default_data(idx,:);

%Downsampling of training data
default_1_rows=find(train_credit_default_data.defaultPaymentNextMonth=='1');
default_0_rows=find(train_credit_default_data.defaultPaymentNextMonth=='0');
undersampled=default_0_rows(1:length(default_1_rows));
rowsToExtract=sort([undersampled;default_1_rows]);
train_credit_default_data=train_credit_default_data(rowsToExtract,:);

%Data split into features and labels
train_credit_default_labels=train_credit_default_data(:,24);
train_credit_default_features=train_credit_default_data(:,1:23);

test_credit_default_labels=test_credit_default_data(:,24);
test_credit_default_features=test_credit_default_data(:,1:23);

%Modeling different classifiers

%using gaussian naive bayesian classifier
distName_Gau={'normal','mvmn','mvmn','mvmn','normal','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn'...
    'normal','normal','normal','normal','normal','normal','normal','normal','normal','normal','normal','normal'};
nbGau = fitcnb(train_credit_default_features,train_credit_default_labels,'ClassNames',{'0'  '1'},'DistributionNames', distName_Gau,'Prior',[0.78 0.22]);
nbGauResult=predict(nbGau,test_credit_default_features);
[prednbGau, PosteriornbGau]=predict(nbGau,test_credit_default_features);

%kfold-10 cross validation for gaussian naive bayesian classifier
nbGauCV = crossval(nbGau);
L=kfoldLoss(nbGauCV);
fprintf('\n10 fold cross valdiation loss for Gaussian naive bayesian classifier:%f',L);

%using naive bayesian classifier kernel density
distNames = {'kernel','mvmn','mvmn','mvmn','kernel','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn'...
    'kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel'};
    % Kernel Width : affects the shape of the kernel used in density estimation
width = 90;
nbKD = fitcnb(train_credit_default_features,train_credit_default_labels,'ClassNames',{'0'  '1'},'DistributionNames', distNames,'width', width,'Prior',[0.78 0.22]);
nbKDResult=predict(nbKD,test_credit_default_features);
[prednbKD, PosteriornbKD]=predict(nbKD,test_credit_default_features);

%kfold-10 cross validation for naive bayesian classifier with kernal density

nbKDCV = crossval(nbKD);
L=kfoldLoss(nbKDCV);
fprintf('\n10 fold cross valdiation loss for Naive bayesian classifier with kernel density:%f',L);

%confusion matrix for naive bayesian classifier
figure(5)
confusionchart(test_credit_default_labels.defaultPaymentNextMonth,categorical(nbGauResult));
confusion_matrix_nbGau=confusionmat(test_credit_default_labels.defaultPaymentNextMonth,categorical(nbGauResult));
title('Confusion Matrix for Gaussian Naive Bayesian Classifier')
[accuracynbGau,precisionnbGau, recallnbGau, specificitynbGau,fscorenbGau] = PerformanceMetrics(confusion_matrix_nbGau);
fprintf('\n\nPerformance Metrics for Gaussian Naive Bayes\n')
fprintf('Accuracy nbGau : %f\n',accuracynbGau)
fprintf('Precision nbGau : %f\n',precisionnbGau)
fprintf('Recall nbGau: %f\n',recallnbGau)
fprintf('Specifcity nbGau : %f\n',specificitynbGau)
fprintf('F1 score nbGau : %f\n',fscorenbGau)


figure(6)
confusionchart(test_credit_default_labels.defaultPaymentNextMonth,categorical(nbKDResult));
confusion_matrix_nbKD=confusionmat(test_credit_default_labels.defaultPaymentNextMonth,categorical(nbKDResult));
title('Confusion Matrix for Kernel Naive Bayesian Classifier')
[accuracynbKD,precisionnbKD, recallnbKD, specificitynbKD,fscorenbKD] = PerformanceMetrics(confusion_matrix_nbKD);
fprintf('\nPerformance Metrics for Kernel Naive Bayes\n')
fprintf('Accuracy nbKD : %f\n',accuracynbKD)
fprintf('Precision nbKD : %f\n',precisionnbKD)
fprintf('Recall nbKD: %f\n',recallnbKD)
fprintf('Specifcity nbKD : %f\n',specificitynbKD)
fprintf('F1 score nbKD : %f\n',fscorenbKD)


%barchart for comparing the measurement of NB

figure(2)
vals=[accuracynbGau accuracynbKD; precisionnbGau precisionnbKD; recallnbGau recallnbKD; specificitynbGau specificitynbKD; fscorenbGau fscorenbKD];
b= bar(vals);
title('Performance measurements for NB')
names = {'accuracy','precision', 'recall', 'specificity','fscore'};
xticklabels(names)
%xtickangle(30)
ylim([0,1])
legend('Gaussian NB', 'Kernel NB')

%barchart for comparing the measurement of NB & RF


%hold on