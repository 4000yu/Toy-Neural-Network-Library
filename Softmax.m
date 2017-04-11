classdef Softmax < handle
    properties (Access = public)
        X
        p
    end
    methods
        function loss = forward(obj, X, Y)
            obj.X = X;
            % For numerical instability
            max_X = max(X,[],2);
            max_X = repmat(max_X,1,size(X,2));
            X = X - max_X;
            
            sum_X = sum(exp(X),2);
            sum_X = repmat(sum_X,1,size(X,2));
            
            obj.p = exp(X) ./ sum_X;
            
            p_label = sum(obj.p.*Y,2);
            loss = sum(-log(p_label));
            loss = loss/size(X,1);            
        end
        function dX = backward(obj, Y)
            dX = obj.p - Y;
        end
    end
end 