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

%% Shuffle data

idx = randperm(size(Xtr,1));
Xtr = Xtr(idx,:);
Ytr = Ytr(idx,:);

%% Define Neural Net:

numClasses = 10;
featureSize = size(Xtr,2);
batchSize = 128;

hiddenLayerSizes = [500 500];


NN = struct();
NN.layers{1} = FullyConnected(sqrt(1/batchSize)*randn(featureSize,hiddenLayerSizes(1)),zeros(1,hiddenLayerSizes(1)));
NN.layers{2} = Tanh();
NN.layers{3} = FullyConnected(sqrt(1/hiddenLayerSizes(1))*randn(hiddenLayerSizes(1),hiddenLayerSizes(2)),zeros(1,hiddenLayerSizes(2)));
NN.layers{4} = Tanh();
NN.layers{5} = FullyConnected(sqrt(1/hiddenLayerSizes(2))*randn(hiddenLayerSizes(2),numClasses),zeros(1,numClasses));
NN.layers{6} = Softmax();


%% Gradient Descent:

opts.lr = 0.1;
opts.reg = 0.01;
opts.type = 'momentum';
opts.momentum = 0.9;
numEpochs = 2;                                                                      

count=1;

for epochNum = 1:numEpochs

    for i=1:batchSize:size(Xtr,1)
        
        if i+batchSize < 50000
            Xtr_batch = Xtr(i:i+batchSize-1,:);
            Ytr_batch = Ytr(i:i+batchSize-1,:);
        else
            Xtr_batch = Xtr(i:end,:);
            Ytr_batch = Ytr(i:end,:);
        end

            [NN,loss] = feedForward(NN,Xtr_batch,Ytr_batch);
            lossTr(count) = loss; loss
            nnops = NN.layers{end}.p; 
            [~,tmp1] = max(nnops,[],2);
            [~,tmp2] = max(Ytr_batch,[],2);
            accTr(count) = sum(tmp1 == tmp2)/size(nnops,1);


            NN = backprop(NN,Ytr_batch);
            NN = updateWeights(NN,opts);

            [NN,loss] = feedForward(NN,Xval,Yval);
            lossVal(count) = loss; 
            nnops = NN.layers{end}.p;
            [~,tmp1] = max(nnops,[],2);
            [~,tmp2] = max(Yval,[],2);
            accVal(count) = sum(tmp1 == tmp2)/size(nnops,1);


            [NN,loss] = feedForward(NN,Xts,Yts);
            lossTs(count) = loss; 
            nnops = NN.layers{end}.p;
            [~,tmp1] = max(nnops,[],2);
            [~,tmp2] = max(Yts,[],2);
            accTs(count) = sum(tmp1 == tmp2)/size(nnops,1);
            
            count = count + 1;

    end
    disp(epochNum);
end







