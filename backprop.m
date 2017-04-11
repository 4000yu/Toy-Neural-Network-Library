function NN = backprop(NN,Y)
    grad = NN.layers{end}.backward(Y);
    for k = length(NN.layers)-1:-1:1
        grad = NN.layers{k}.backward(grad);
    end
end