function [NN,loss]  = feedForward(NN,X,Y)
    activation = X;
    for k = 1:length(NN.layers)-1
        activation = NN.layers{k}.forward(activation);
    end
    loss = NN.layers{end}.forward(activation,Y);
end