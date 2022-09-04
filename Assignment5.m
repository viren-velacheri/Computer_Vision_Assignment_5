% Assignment 5
% Name: Viren Velacheri, EID: vv6898
% Slip days: 0

% This is just the set up where I am loading the batch files from my 
% computer. Specifically I am getting the data files as well as the labels

% Note: I just implemented the K Nearest Neighbor Classifier. I did not
% have the time to do the adaboost and other following stuff as I had run
% out of slip days. Also, the overall program takes 17 seconds to run as it
% includes all the preprocessing before the knnsearch algorithm is run.

firstDataBatch = load('C:\Users\virenv\Documents\Computer_Vision_Matlab_Code\Assignment_5\cifar-10-batches-mat\data_batch_1.mat', 'data');
firstDataBatchLabels = load('C:\Users\virenv\Documents\Computer_Vision_Matlab_Code\Assignment_5\cifar-10-batches-mat\data_batch_1.mat', 'labels');

secondDataBatch = load('C:\Users\virenv\Documents\Computer_Vision_Matlab_Code\Assignment_5\cifar-10-batches-mat\data_batch_2.mat', 'data');
secondDataBatchLabels = load('C:\Users\virenv\Documents\Computer_Vision_Matlab_Code\Assignment_5\cifar-10-batches-mat\data_batch_2.mat', 'labels');

thirdDataBatch = load('C:\Users\virenv\Documents\Computer_Vision_Matlab_Code\Assignment_5\cifar-10-batches-mat\data_batch_3.mat', 'data');
thirdDataBatchLabels = load('C:\Users\virenv\Documents\Computer_Vision_Matlab_Code\Assignment_5\cifar-10-batches-mat\data_batch_3.mat', 'labels');

fourthDataBatch = load('C:\Users\virenv\Documents\Computer_Vision_Matlab_Code\Assignment_5\cifar-10-batches-mat\data_batch_4.mat', 'data');
fourthDataBatchLabels = load('C:\Users\virenv\Documents\Computer_Vision_Matlab_Code\Assignment_5\cifar-10-batches-mat\data_batch_4.mat', 'labels');

fifthDataBatch = load('C:\Users\virenv\Documents\Computer_Vision_Matlab_Code\Assignment_5\cifar-10-batches-mat\data_batch_5.mat', 'data');
fifthDataBatchLabels = load('C:\Users\virenv\Documents\Computer_Vision_Matlab_Code\Assignment_5\cifar-10-batches-mat\data_batch_5.mat', 'labels');

testBatch = load('C:\Users\virenv\Documents\Computer_Vision_Matlab_Code\Assignment_5\cifar-10-batches-mat\test_batch.mat', 'data');
testBatchLabels = load('C:\Users\virenv\Documents\Computer_Vision_Matlab_Code\Assignment_5\cifar-10-batches-mat\test_batch.mat', 'labels');

%  Here I convert all the data and labels to actual matrices since
% I saw that the data batches and labels are structures and so 
% I do this to convert them to matrices.
firstDataBatch = cell2mat(struct2cell(firstDataBatch));
firstDataBatch = double(firstDataBatch);
firstDataBatchLabels = cell2mat(struct2cell(firstDataBatchLabels));
firstDataBatchLabels = double(firstDataBatchLabels);

secondDataBatch = cell2mat(struct2cell(secondDataBatch));
secondDataBatch = double(secondDataBatch);
secondDataBatchLabels = cell2mat(struct2cell(secondDataBatchLabels));
secondDataBatchLabels = double(secondDataBatchLabels);

thirdDataBatch = cell2mat(struct2cell(thirdDataBatch));
thirdDataBatch = double(thirdDataBatch);
thirdDataBatchLabels = cell2mat(struct2cell(thirdDataBatchLabels));
thirdDataBatchLabels = double(thirdDataBatchLabels);

fourthDataBatch = cell2mat(struct2cell(fourthDataBatch));
fourthDataBatch = double(fourthDataBatch);
fourthDataBatchLabels = cell2mat(struct2cell(fourthDataBatchLabels));
fourthDataBatchLabels = double(fourthDataBatchLabels);

fifthDataBatch = cell2mat(struct2cell(fifthDataBatch));
fifthDataBatch = double(fifthDataBatch);
fifthDataBatchLabels = cell2mat(struct2cell(fifthDataBatchLabels));
fifthDataBatchLabels = double(fifthDataBatchLabels);

testBatch = cell2mat(struct2cell(testBatch));
testBatch = double(testBatch);
testBatchLabels = cell2mat(struct2cell(testBatchLabels));
testBatchLabels = double(testBatchLabels);

% The below lines merge the data batches together as well as the labels
% together in a row wise manner.
totalDataBatch = [firstDataBatch; secondDataBatch; thirdDataBatch; fourthDataBatch; fifthDataBatch];
totalDataBatchLabels = [firstDataBatchLabels; secondDataBatchLabels; thirdDataBatchLabels; fourthDataBatchLabels; fifthDataBatchLabels];
% Subtract the column mean from the total data to center the data.
% Do the same thing for the test batch data.
totalDataBatch = totalDataBatch - mean(totalDataBatch, 1);
testBatch = testBatch - mean(testBatch, 1);
% This is the covariance matrix that is generated
coVariance = totalDataBatch' * totalDataBatch;

% The eigenvalues and eigenvectors are computed based on this covariance
% matrix
[eigenvectors, eigenvalues] = eig(coVariance);

% The eigenvectors are sorted in descending order based on the 
% eigenvalues
[~, indices] = sort(diag(eigenvalues), 'descend');
eigenvectors = eigenvectors(:, indices);

% Projection matrix is computed based on the first 100 eigenvectors 
% and then used for the dimensional reduction for both data and test
% batch matrices.
projectionMatrix = eigenvectors(:, 1:100);
newDataMatrix = totalDataBatch * projectionMatrix;
newTestBatch = testBatch * projectionMatrix;

% Get the indices of the k nearest neighbors
[mIdx,~] = knnsearch(newDataMatrix,newTestBatch,'K',10,'Distance','euclidean');

nearestNeighbors=mIdx(:,1:10);

predictedLabels = [];

% A simple set of for loops that gets the majority vote for labels.
% This is how the labels are predicted.
i = 1;
while i <= size(nearestNeighbors,1)
    uniqueDataBatchLabelVals = unique(totalDataBatchLabels(nearestNeighbors(i,:)'));
    maxCount = 0;
    maxLabel = 0;
    j = 1;
    while j <= length(uniqueDataBatchLabelVals)
        matches = find(totalDataBatchLabels(nearestNeighbors(i,:)') == uniqueDataBatchLabelVals(j));
        count = length(matches);
        if count >= maxCount
            maxLabel = uniqueDataBatchLabelVals(j);
            maxCount = count;
        end
        j = j + 1;
    end
    i = i + 1;
    predictedLabels= [predictedLabels maxLabel];
end

predictedLabels = predictedLabels';

% This is the accuracy that is calculated as well as displayed.
accuracy = length(find(predictedLabels==testBatchLabels))/size(newTestBatch,1);
accuracy 

% After initiliazing confusion matrix as a 10 by 10 matrix of zeros
% , simply go through this simple for loop to populate it accordingly.
% The confusion matrix is then displayed at the end as well.
confusion_mat = zeros(10, 10);
for i=1:size(predictedLabels,1)
    confusion_mat(predictedLabels(i) + 1, testBatchLabels(i) + 1) = confusion_mat(predictedLabels(i) + 1, testBatchLabels(i) + 1) + 1;
end

confusion_mat