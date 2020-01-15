% Out of bag error estimates to determine range of values for number of
% trees and minimum number of leaves . OOB error estimates for number of
% grown trees takes some time to run.

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


%Data split into features and labels
train_credit_default_labels=train_credit_default_data(:,24);
train_credit_default_features=train_credit_default_data(:,1:23);

test_credit_default_labels=test_credit_default_data(:,24);
test_credit_default_features=test_credit_default_data(:,1:23);

%Modeling using random forest classfier
%Selected number of trees as 300 to observe OOB for all grown trees till it
treeRandomForest = TreeBagger(300,train_credit_default_features,train_credit_default_labels,'Prior',[0.78 0.22],'OOBPredictorImportance','On');

%out of box classification error to observe the values based on number of
%trees grown
figure
plot(oobError(treeRandomForest))
xlabel('Number of Grown Trees')
ylabel('Classification Error')
title('Classification Error v/s Number of trees')


%Selected range of minimum number of leaves
leaf = [1 5 10 25 50 75 100];
%Based on previous excercise around 100 trees gave minimum error
nTrees = 100;
rng(1234,'twister');
savedRng = rng; % save the current RNG settings
color = 'bgr';
for leaves = 1:length(leaf)
   % Reinitialize the random number generator, so that the
   % random samples are the same for each leaf size
   rng(savedRng)
   % Create a bagged decision tree for each leaf size and plot out-of-bag
   % error 'oobError'
   b = TreeBagger(nTrees,train_credit_default_features,train_credit_default_labels,'OOBPrediction','on',...
                         'MinLeafSize',leaf(leaves));
   plot(oobError(b))
   hold on
end
xlabel('Number of grown trees')
ylabel('Out-of-bag classification error')
legend({'1', '5', '10','25','50','100'},'Location','NorthEast')
title('Classification Error for Different Leaf Sizes')
hold off