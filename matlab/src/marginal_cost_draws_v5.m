function [type_space,marg_distrib] = marginal_cost_draws_v5(mu,sigma2,s)

%% Type Space
% Initialize Type Space
type_space = cell(length(mu),1);

% Type Space
for ind=1:length(mu)
%   maxind = min( floor(3*sqrt(sigma2(ind,ind))/step_size(ind)), floor(mu(ind)/step_size(ind)) );
%    type_space{ind,1} = mu(ind) + step_size(ind)*[-maxind:maxind,maxind*1.2]';
    type_space{ind,1} = linspace(0,6,s)';

end



%% Marginal Distribution

marg_distrib = pdf('normal',type_space{ind,1},mu(1),sigma2(1,1));
marg_distrib = marg_distrib./sum(marg_distrib,1);

end

