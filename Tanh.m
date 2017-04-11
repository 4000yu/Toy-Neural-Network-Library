classdef Tanh < handle
    properties (Access = public)
        X
    end
    methods
        function Y = forward(obj, X)
            obj.X = X;
            Y = (1.7159)*tanh((2/3)*X);
        end
        function dX = backward(obj, dY)
            dX = (2/3)*1.7159*( 1 - (tanh((2/3)*obj.X)).^2 ).* dY;
        end
    end
end 