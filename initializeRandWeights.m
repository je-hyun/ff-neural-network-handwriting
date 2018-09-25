function thetas = initializeRandWeights(nn_arch)
  %% Return a randomly initialized cell array of weights, thetas.
  %  The size and number of thetas is based on the input array of layer sizes.
  %  The range of each value is (-init_epsilon, init_epsilon)
  %% Example: initializeRandWeights([1,3,2,7]) -> {[3x2],[2x4],[7x3]}
  %  Each theta is sized s(L + 1) x (s(L) + 1),
  %  where s(L) is the number of units in layer L.
  
  %initialize thetas as cell array
  thetas = {};
  
  % range of weight randomization
  init_epsilon = 0.12;
  
  % loop through number of thetas and set between -init_epsilon to init_epsilon
  for i = 1 : length(nn_arch)-1
    thetas{i} = rand(nn_arch(i + 1), nn_arch(i) + 1) * 2 * init_epsilon - init_epsilon;
  endfor
end