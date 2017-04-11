classdef FullyConnected < handle
    properties (Access = public)
        W
        b
        vW
        vb
        dW
        db
        X
    end
    methods
        function obj = FullyConnected(W, b)
            obj.W = W;
            obj.b = b;
        end
        function Y = forward(obj, X)
            obj.X = X;
            tmp = ones(size(X,1),1);
            tmp = tmp * obj.b;
            Y = X*obj.W + tmp;
        end
        function dX = backward(obj, dY)
            obj.dW = (obj.X' * dY)/size(obj.X,1);
            obj.db = sum(dY)/size(obj.X,1);
            dX = dY * obj.W';
        end
    end
end 