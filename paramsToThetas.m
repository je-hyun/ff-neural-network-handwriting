function thetas = paramsToThetas(params, nn_arch)
  %% Return cellArray of thetas with dimensions defined by nn_arch
  %  with each matrix representing theta(Layer).
  thetas = {};
  begin_count = 1;
  for i = 1 : length(nn_arch) - 1
    this_size = nn_arch(i) + 1;
    next_size = nn_arch(i+1);
    thetas{i} = reshape(params(begin_count:begin_count + next_size * this_size - 1),next_size,this_size);
    begin_count += next_size * this_size;
  endfor
end