% Compare performance between the tuned naive bayes and random forest
% models on a test set

%Clear workspace, command window and close figures
clear all;
clc;
close all;

%Dataloading
%Selecting specific rows range as the original data set has 2 header rows and a
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

%Marriage has no value '0' based on the data definition hence assigning to
%a defined value '3' for others
credit_default_data.MARRIAGE(strcmpi(credit_default_data.MARRIAGE,'0')) = {'3'};

%Feature scaling and normalization of numeric features  to z-score of the data with center 0 and standard deviation 1.
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

%Downsampling of training data to ensure that both the classifer labels
%are represented in equal proportion
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

%Data split into features and labels
train_credit_default_labels=train_credit_default_data(:,24);
train_credit_default_features=train_credit_default_data(:,1:23);

test_credit_default_labels=test_credit_default_data(:,24);
test_credit_default_features=test_credit_default_data(:,1:23);


%Modeling different classifiers
%Using random forest
%Number of trees, numbe rof predictors to sample and minimum number of leaves selected based on hyper parameter tuning
%results
treeRandomForest = TreeBagger(80,train_credit_default_features,train_credit_default_labels,'ClassNames',{'0'  '1'},'Prior',[0.78 0.22],'MinLeafSize',50,'Method','classification','NumPredictorsToSample',10);
treeRandomForestResult=predict(treeRandomForest,test_credit_default_features);
[predRF, PosteriorRF]=predict(treeRandomForest,test_credit_default_features);
%confusion matrix for random forest classifier
figure(4)
confusionchart(test_credit_default_labels.defaultPaymentNextMonth,categorical(treeRandomForestResult));
confusion_matrix_RF=confusionmat(test_credit_default_labels.defaultPaymentNextMonth,categorical(treeRandomForestResult));
%calculating the perfromance metrics for random forest classifier
[accuracyRF,precisionRF, recallRF, specificityRF,fscoreRF] = PerformanceMetrics(confusion_matrix_RF);
fprintf('Performance Metrics for Random Forest\n')
fprintf('Accuracy RF : %f\n',accuracyRF)
fprintf('Precision RF : %f\n',precisionRF)
fprintf('Recall RF : %f\n',recallRF)
fprintf('Specificity RF : %f\n',specificityRF)
fprintf('F1 score RF : %f\n',fscoreRF)

title('Confusion Matrix for Random Forest Classifier')

%using kernel based naive bayesian classifier as based on hyper parameter
%tuning it gave better F1 score
%naiveBayes = fitcnb(train_credit_default_features,train_credit_default_labels);

distNames = {'kernel','mvmn','mvmn','mvmn','kernel','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn'...
    'kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel'};
width = 90;

naiveBayes = fitcnb(train_credit_default_features,train_credit_default_labels,'ClassNames',{'0'  '1'},'DistributionNames', distNames,'width', width,'Prior',[0.78 0.22]);
naiveBayesResult=predict(naiveBayes,test_credit_default_features);
[predNB, PosteriorNB]=predict(naiveBayes,test_credit_default_features);
%confusion matrix for naive bayesian classifier
figure(5)
confusionchart(test_credit_default_labels.defaultPaymentNextMonth,categorical(naiveBayesResult));
confusion_matrix_NB=confusionmat(test_credit_default_labels.defaultPaymentNextMonth,categorical(naiveBayesResult));
[accuracyNB,precisionNB, recallNB, specificityNB,fscoreNB] = PerformanceMetrics(confusion_matrix_NB);
fprintf('Performance Metrics for Naive Bayes\n')
fprintf('Accuracy NB : %f\n',accuracyNB)
fprintf('Precision NB : %f\n',precisionNB)
fprintf('Recall NB : %f\n',recallNB)
fprintf('Specifcity NB : %f\n',specificityNB)
fprintf('F1 score NB : %f\n',fscoreNB)
title('Confusion Matrix for Naive Bayesian Classifier')


%compare performance between naive bayes and random forest
% ROC curve
[XNB, YNB, TNB, AUCNB] = perfcurve(test_credit_default_labels.defaultPaymentNextMonth, PosteriorNB(:, 1), '0');
[XRF, YRF, TR, AUCRF] = perfcurve(test_credit_default_labels.defaultPaymentNextMonth, PosteriorRF(:, 1), '0');
figure(6)
plot(XNB, YNB)
hold on
plot(XRF, YRF)
pause(0.1);
legend('Naive Bayes', 'Random Forest')
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('ROC Curve Comparison : Naive Bayes vs Random Forest')
hold off

%Performance Metrics
figure(7)
vals=[accuracyNB accuracyRF; precisionNB precisionRF; recallNB recallRF; specificityNB specificityRF; fscoreNB fscoreRF];
b= bar(vals);
%b(3).FaceColor = [.6 .5 .2];
title('Performance comparison between Tuned NB and RF')
names = {'accuracy','precision', 'recall', 'specificity','fscore'};
xticklabels(names)
%xtickangle(30)
ylim([0,1])
legend('Naive Bayes', 'Random Forest')
%hold on