%% initialize data and useful variables
load('dataset.mat');

% initialize options for NN
architecture = [400, 25, 10];
lambda = 1;
options = optimset('MaxIter', 50);

%% Minimize J with respect to thetas.
%  Randomly initialize all weights.
init_thetas = initializeRandWeights(architecture);
init_params = thetasToParams(init_thetas);
%  costFunction is a lambda function so we can pass as arg
costFunction = @(p) nnCostFunction(p, architecture, X, y, lambda);

fprintf('Ready to train. (press enter to continue)\n');
pause;

[nn_params, cost] = fmincg(costFunction, init_params, options);

final_thetas = paramsToThetas(nn_params, architecture);

% predict / check results
my_prediction = predict(X, final_thetas);
my_training_accuracy = mean(double(my_prediction==y)) * 100;
fprintf('Accuracy of NN: %.2f%%\n',my_training_accuracy);