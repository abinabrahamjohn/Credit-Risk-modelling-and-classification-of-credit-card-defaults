%Performing Exploratory Data analysis for the dataset
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

%Exploratory data analysis
%Determining the overall distribution of the target in dataset
labels = {'0','1'};
figure(1)
pie(credit_default_data.defaultPaymentNextMonth)
title('Target Class Imbalance')

fprintf('**Exploratory Data Analysis on Credit card Default**\n')
%Determing the dimensions of the dataset
[m n] = size(credit_default_data);
fprintf('The dataset has %d Rows and %d Columns.\n', m, n)
fprintf('The dataset has %d Numeric and %d Categorical attributes.\n',...
    n-length(catColumns), length(catColumns))
%Summarising the daat set min, max, median
summary(credit_default_data)

%Histogram
% Univariate Analysis : Histograms of Features
figure('pos',[10 10 1000 600])
title('Histogram of features')
for col_index = 1:n-1
   subplot(5,5,col_index)
   histogram(credit_default_data{:, col_index})
   title(sprintf('Histogram of %s', credit_default_data.Properties.VariableNames{col_index}))
end

% Computing the Correlation Matrix (Pearson coefficient) for only
% numeric features
figure(3)
corrMatrix = corr(table2array(credit_default_data(:, ~catColumnsFilter)),'type','Pearson');
% Plot the Correlation Matrix
xvalues = {'LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT15','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'};
yvalues = {'LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT15','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'};
title('Correlation Chart')
heatmap(xvalues,yvalues,corrMatrix);
title('Correlation Plot (Pearson) - Numeric Attributes')
%Observe that there is high collinearity between all the Bill Amounts. Can
%be removed before modeling