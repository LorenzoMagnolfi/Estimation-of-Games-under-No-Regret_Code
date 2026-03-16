function [mc_draws] = marginal_cost_draws_v4(type_space,draws)
global sigma2 mu

%% Generate (Double) Truncated `Discrete' Normal Distribution
% For 2 dimensions, use the mutlivariate pdf values
if size(sigma2,1) == 2
    
    % All Possible Type Combinations (Using Kronecker Product) 
    combs = [kron(ones(size(type_space{2,1})),type_space{1,1}), kron(type_space{2,1},ones(size(type_space{1,1})))];
    
    % Probabilities of Combinations Using Multivariate Normal pdf
    probs = mvnpdf(combs,mu',sigma2');
    
    % Marginal Cost Indices
    inds = (1:size(combs,1))';
    
    % Random Draws from Marginal Cost Combinations
    mc_inds = randsample(inds,draws,true,probs);
    mc_draws = combs(mc_inds,:);
     
% For independently distributed marginal costs in greater than three 
% dimensions use the univariate pdf values one dimension at a time
elseif sigma2 == diag(diag(sigma2)) 
    % 
    for ind=1:size(sigma2,1)
        % Probabilities of Combinations Using Multivariate Normal pdf
        probs = mvnpdf(type_space{ind,1},mu(ind),sigma2(ind,ind));
    
        % Marginal Cost Indices
        mc_draws(:,ind) = randsample(type_space{ind,1},draws,true,probs);
        
    end
    
% For arbitrary dependence in multiple dimensions, use the true continuous 
% multidemsional joint distribution, and lump results into bins 
else
    % Multiplier
    mult = length(mu)+1;

    % Discrete Random Normal
    discrand = mu'+step_size'.*round(mvnrnd(zeros(size(mu)),sigma2,1000+mult*draws)./step_size');

    % Truncate Distribution Tails
    bad_inds = sum(abs(discrand-mu')>3*sqrt(diag(sigma2))',2);
    discrand(bad_inds>0,:) = [];

    % Marginal Cost Draws
    mc_draws = discrand(1:draws,:);

end
