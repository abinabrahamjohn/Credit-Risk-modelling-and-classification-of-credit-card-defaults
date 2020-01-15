% Function to calculate the key visualization metrics- accuracy, precision,
% recall, specificity, f1score
function [accuracy,precision, recall, specificity,f1score] = PerformanceMetrics(confusion_matrix)
%Accuracy: (TP+TN)/(TP+TN+FP+FN)
accuracy = (confusion_matrix(1,1) + confusion_matrix(2,2)) / sum(sum(confusion_matrix));

%Precision: TP/(TP+FP)
precision = confusion_matrix(1,1) / (confusion_matrix(1,1) + confusion_matrix(2,1));

%Recall: TP/ (TP+FN)
recall = confusion_matrix(1,1) / (confusion_matrix(1,1) + confusion_matrix(1,2));

%Specificity: TN/(TN+FP )
specificity = confusion_matrix(2,2) / (confusion_matrix(2,2) + confusion_matrix(2,1));

%F1Score: 2*Precision*Recall/(Recall+Precision)
f1score = 2 * precision * recall / (precision + recall);
end

