function NN = updateWeights(NN, opts)
    for k = 1:length(NN.layers)
        if strcmp(class(NN.layers{k}),'FullyConnected')
            %% Add L2 regularization
            if exist('opts.reg')
                NN.layers{k}.dW = NN.layers{k}.dW + opts.reg*NN.layers{k}.W;
            end
            
            %% Vanilla (just LR)
            if strcmp(opts.type,'vanilla')
                NN.layers{k}.W = NN.layers{k}.W - opts.lr*NN.layers{k}.dW;
                NN.layers{k}.b = NN.layers{k}.b - opts.lr*NN.layers{k}.db;
            end

            %% Momentum
            if strcmp(opts.type, 'momentum')
                if isempty(NN.layers{k}.vW)
                    NN.layers{k}.vW = - opts.lr*NN.layers{k}.dW;
                    NN.layers{k}.vb = - opts.lr*NN.layers{k}.db;
                else
                    NN.layers{k}.vW = opts.momentum*NN.layers{k}.vW - opts.lr*NN.layers{k}.dW;
                    NN.layers{k}.vb = opts.momentum*NN.layers{k}.vb - opts.lr*NN.layers{k}.db;
                end
                NN.layers{k}.W = NN.layers{k}.W + NN.layers{k}.vW;
                NN.layers{k}.b = NN.layers{k}.b + NN.layers{k}.vb;
            end

            %% Nesterov
            if strcmp(opts.type, 'nesterov')
            end
        end
    end

end