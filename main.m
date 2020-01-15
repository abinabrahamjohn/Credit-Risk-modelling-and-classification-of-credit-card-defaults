%Main script to run between all options

%Clear workspace, command window and close figures
clear all;
clc;
close all;

fprintf('\nWelcome to the Machine Learning Coursework!\n');
while 1
    fprintf('\nPlease select a choice between 1 and 6:\n\n')
    fprintf('1 : Exploratory Data Analysis\n')
    fprintf('2 : Feature Importance using MSME algorithm\n')
    fprintf('3 : Random Forest - Grid Search for hyperparamter tuning(Note- script takes few minutes)\n')
    fprintf('4 : Naive Bayes - Hyper parameter tuning\n')
    fprintf('5 : Tuned Model performance metrics\n')
    fprintf('6 : Random Forest - Exploring hyperparameter range using OOB error estimate(Note- script takes few minutes)\n')
    fprintf('Type 0 to exit the program ...\n\n')
    choice = input('Enter choice: ');
    switch choice
        case 1
            % Script : Exploratory Data Analysis
            EDA;
            pause(2)
        case 2
            FeatureImportance;
            pause(2)
        case 3
            GridSearchRF;
            pause(2)
        case 4
            HyperParameterTuningNB;
            pause(2)
        case 5
            MLCourseworkTunedModelFS;
            pause(2)
        case 6
            HyperParameterOptionRF;
            pause(2)
        case 0
            run=0;
            break;
        otherwise
            fprintf('\nPlease select a value from the given choice\n')
            pause(1)
    end

end
fprintf('\nThank you. Have a great day!\n');
fprintf('\n-Abin Abraham and Soyeon Park\n');
pause(5)
% Clear workspace, console
close all; clear all; clc;
