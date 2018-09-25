function [J grad] = nnCostFunction(nn_params, ...
                                   nn_arch, ...
                                   X, y, lambda)
%% Return the cost and gradient for a set of params.

% get thetas from params
thetas = paramsToThetas(nn_params,nn_arch);

% init some useful variables:
m = size(X,1);
num_labels = nn_arch(end);
grad_thetas = {};
L = length(nn_arch);
grad = [];


for i = 1:L-1
  grad_thetas{i} = zeros(size(thetas{i}));
endfor

%% Fprop
my_A = X;
for i = 1:L-1
  my_A = sigmoid([ones(m,1) my_A] * thetas{i}');
endfor

% set hot_y as hot-encoded y e.g. [[1, 0, 0, ... , 0]; ...]
hot_y = zeros(m,num_labels);
temp_index = sub2ind (size(hot_y), 1:rows(hot_y), y');
hot_y(temp_index) = 1;

%% cost function J without regularization
J = (-1/m) * sum((hot_y .* log(my_A) + (1 - hot_y) .* log(1 - my_A))(:));

% add regularization term
reg_term = 0;
for i = 1:L-1
  reg_term += sum((thetas{i}(:,2:end).^2)(:));
endfor
reg_term = (lambda/(2*m)) * reg_term;
J += reg_term;

%% Backprop
for i = 1:m
  % working with example i
  x = X(i,:)';
  %% fprop
  % cellArrays a and z for each layer
  a = {};
  z = {};
  a{1} = [1;x];
  % fprop, saving every a and z.
  for j = 1:L-1
    z{j+1} = thetas{j} * a{j};
    a{j+1} = sigmoid(z{j+1});
    % don't add bias term to output layer
    if j != L-1
      a{j+1} = [1;a{j+1}];
    endif
  endfor
  deltas = {};
  deltas{L} = a{L} - hot_y(i,:)';
  % calculate deltas backwards from L-1 to 2
  for k = L-1:-1:2
    % TODO: check if this actually works.
    deltas{k} = (thetas{k}'*deltas{k+1})(2:end) .* sigmoidGradient(z{k});
  endfor
  % calculate grad_thetas
  for j = 1:L-1
    grad_thetas{j} += deltas{j+1} * a{j}';
  endfor
endfor

% finally calculate grad (by scaling by 1/m and regularizing non-bias terms)
for k = 1:L-1
  grad_thetas{k} = (1/m) * grad_thetas{k};
  grad_thetas{k} = [grad_thetas{k}(:,1), grad_thetas{k}(:,2:end) + (lambda/m) * thetas{k}(:,2:end)];
  grad = [grad ; grad_thetas{k}(:)];
endfor

end