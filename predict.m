function p = predict(X, thetas)
  % Return a prediction of y based on X and thetas.
  
  % useful values
  m = size(X, 1);
  my_A = X;
  
  % calculate prediction by iterating thetas
  for cur_theta = 1:length(thetas)
    my_A = sigmoid([ones(m,1) my_A] * thetas{cur_theta}');
  endfor
  [~, p] = max(my_A, [], 2);
end