classdef Sigmoid < handle
    properties (Access = public)
        Y
    end
    methods
        function Y = forward(obj, X)
            Y = 1.0 ./ (1.0 + exp(-X));
            obj.Y = Y;
        end
        function dX = backward(obj, dY)
            dX = obj.Y .* (1 - obj.Y) .* dY;
        end
    end
end 