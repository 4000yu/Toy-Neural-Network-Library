clear;
clc; close all;
%% Data Handling:

imagesTr = loadMNISTImages('train-images.idx3-ubyte');
labelsTr = loadMNISTLabels('train-labels.idx1-ubyte')+1;
imagesVal = double(imagesTr(:,50001:60000)');
imagesTr = double(imagesTr(:,1:50000)');
labelsVal = labelsTr(50001:60000);
labelsTr = labelsTr(1:50000);
imagesTs = loadMNISTImages('t10k-images.idx3-ubyte')';
labelsTs = loadMNISTLabels('t10k-labels.idx1-ubyte')+1;



Xtr =imagesTr - mean(imagesTr,2)*ones(1,784);
Xval =imagesVal - mean(imagesVal,2)*ones(1,784);
Xts =imagesTs - mean(imagesTs,2)*ones(1,784);

tmp = eye(10);
Ytr = tmp(labelsTr,:);
Yts = tmp(labelsTs,:);
Yval = tmp(labelsVal,:);

%% Define Neural Net:

numClasses = 10;
featureSize = size(Xtr,2);

hiddenLayerSizes = 500;


NN = struct();
NN.layers{1} = FullyConnected(0.1*randn(featureSize,hiddenLayerSizes(1)),zeros(1,hiddenLayerSizes(1)));
%%NN.layers{1} = FullyConnected(0.01*randn(featureSize,numClasses),zeros(1,numClasses));
NN.layers{2} = Sigmoid();
NN.layers{3} = FullyConnected(0.1*randn(hiddenLayerSizes(1),numClasses),zeros(1,numClasses));
NN.layers{4} = Softmax();


%% Gradient Descent:

opts.lr = 0.1;
opts.reg = 0.01;
opts.type = 'vanilla';
numEpochs = 500;                                                                      



for epochNum = 1:numEpochs
    
    [NN,loss] = feedForward(NN,Xtr,Ytr);
    lossTr(epochNum) = loss;
    nnops = NN.layers{end}.p; 
    [~,tmp1] = max(nnops,[],2);
    [~,tmp2] = max(Ytr,[],2);
    accTr(epochNum) = sum(tmp1 == tmp2)/size(nnops,1);
    
    
    NN = backprop(NN,Ytr);
    NN = updateWeights(NN,opts);
    
    [NN,loss] = feedForward(NN,Xval,Yval);
    lossVal(epochNum) = loss; 
    nnops = NN.layers{end}.p;
    [~,tmp1] = max(nnops,[],2);
    [~,tmp2] = max(Yval,[],2);
    accVal(epochNum) = sum(tmp1 == tmp2)/size(nnops,1);
    
    
    [NN,loss] = feedForward(NN,Xts,Yts);
    lossTs(epochNum) = loss; 
    nnops = NN.layers{end}.p;
    [~,tmp1] = max(nnops,[],2);
    [~,tmp2] = max(Yts,[],2);
    accTs(epochNum) = sum(tmp1 == tmp2)/size(nnops,1);
    
    fprintf('%f %f\n',epochNum,accVal(epochNum))
    %disp(epochNum); disp(accVal);
end







