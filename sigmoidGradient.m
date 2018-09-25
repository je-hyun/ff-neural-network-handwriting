function g = sigmoidGradient(z)
% Return the gradient of the sigmoid function at input z.
gz = sigmoid(z);
g = gz .* (1 - gz);
end