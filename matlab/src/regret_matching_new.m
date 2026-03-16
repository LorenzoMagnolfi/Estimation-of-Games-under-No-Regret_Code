function [opt_action, R_n] = regret_matching_new(Act,actions_t,mc_t,mc_draw,j,alpha,ind)

%==========================================================================
%FUNCTION: regret_matching
%AUTHOR: Jonathan E. Becker
%DESCRIPTION: Simple implementation of regret matching algorithm for the
%pricing game with arbitrary players, actions, and marginal costs.
%--------------------------------------------------------------------------
%INPUTS:
%1. A | Set of (real-valued) actions available to player j
%2. actions | Matrix of past actions for all players
%3. mc | Matrix of past marginal costs for all players
%4. mc_draw | Vector of marginal costs for all players in current period
%5. j | Scalar index for the current player
%OUTPUTS: 
%1. opt_action : Optimal action in A
%==========================================================================


% For trials:
%Act = AA;
%actions_t = actions(1:N+ind-1,:);
%mc_t = mc(1:N+ind-1,:);
%mc_draw = mc(N+ind,:);

% Past Instance of the same type
type_inds = mc_t(:,j) == mc_draw(j);
type_inds = type_inds(1:size(actions_t,1));

% Utility in each past period
util = choice_prob(actions_t(type_inds,j),actions_t(type_inds,:),j,alpha).*(actions_t(type_inds,j)-mc_t(type_inds,j));
U_n = sum(util)/ind;

% Utility in each past period to playing actions in A against a_not
cf_util = choice_prob(Act',actions_t(type_inds,:),j,alpha).*(Act'-mc_draw(j));

% V_n = max(sum(cf_util)/length(cf_util));
V_n = max(sum(cf_util)/ind);
R_n = max(V_n-U_n,0);

regret = (cf_util-util);

if isnan(regret)
regret_t = zeros(size(Act,1),1);
else
    regret_t  = mean(regret,1);
end

% REWRITE LIKE GAMMA IN THE DRAFT!

% Weights equal to the average utility of playing 
pA = max(regret_t,0)';
if sum(pA) == 0
    pA = pA + 1;
end



% Optimal Action
opt_action = randsample(Act,1,true,pA);
snapnow;
