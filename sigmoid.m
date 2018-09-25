function g = sigmoid(z)
% Return sigmoid function of input z: 1/(1+e^-z)
g = 1.0 ./ (1.0 + exp(-z));
end
