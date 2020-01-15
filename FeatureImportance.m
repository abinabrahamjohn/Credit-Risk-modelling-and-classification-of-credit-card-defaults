%Rank features for classification using minimum redundancy maximum relevance (MRMR) algorithm

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
rng(111)
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
%fprintf('The downsampled training dataset has %d 0s and %d 1s in the target.\n', down_sampled_counts(1), down_sampled_counts(2))

%Data split into features and labels
train_credit_default_labels=train_credit_default_data(:,24);
train_credit_default_features=train_credit_default_data(:,1:23);


test_credit_default_labels=test_credit_default_data(:,24);
test_credit_default_features=test_credit_default_data(:,1:23);


%feature selection
figure
%Rank features for classification using minimum redundancy maximum
%relevance (MRMR) algorithm and returns the index with scores
[idx,scores] = fscmrmr(train_credit_default_features,train_credit_default_labels);
%Create a bar plot of the predictor importance scores.
bar(scores(idx))
xlabel('Predictor rank')
ylabel('Predictor importance score')
%All xticklabels do not seem to be getting displayed properly.Need to fix.
xlab=train_credit_default_features.Properties.VariableNames(idx);
xtickangle(30);
%set(gca, 'XTick', linspace(0,23,length(xlab)), 'XTickLabels', xlab);
xticklabels(strrep(train_credit_default_features.Properties.VariableNames(idx),'_','\_'))
i=idx(1:10);
fprintf('\n**The top 10 most important features based on MSMR algorithm is**\n')
count=1;
for k=i
    fprintf('%d. %s\n',count,cell2mat(train_credit_default_features.Properties.VariableNames(k)))
    count=count+1;
end
