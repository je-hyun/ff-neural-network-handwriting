function params = thetasToParams(thetas)
  % Return flattened vector from values of cell array thetas.
  params = [];
  for i = 1:length(thetas)
    params = [params; thetas{i}(:)];
  endfor
end