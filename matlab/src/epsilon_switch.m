
function eps = epsilon_switch(maxiters,conf,switch_eps)
 
% SwitchEps = 1 --- eps with new conv rates
% SwitchEps = 0 --- fixed eps with new largest dev
% SwitchEps = 2 --- fixed eps with OLD largest dev ONLY WORKS FOR 2
% ACTIONS!
% SwitchEps = 3 --- fixed eps with NEW DEF Feb 2024
% SwitchEps = 4 --- EXP3 bounds
% SwitchEps = 5 --- Stochastic bandit w/ full monitoring

global Pi NAct NPlayers s
if switch_eps <= 1 || switch_eps==3 || switch_eps==4
    Kappa = max(Pi(:,:,1))-(min(Pi(:,:,1))); % will be used to multiply eps!
elseif switch_eps ==2
    Kappa = max(abs(Pi(1:2,:,1)-Pi(3:4,:,1))); 
elseif switch_eps ==5
    for jj = 1:size(Pi,2)
    Kappa(jj) = min(pdist(Pi(:,jj,1), 'euclidean'));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% EPS linked to rate of convergence!
if switch_eps == 1
    %eps = NPlayers*s*Kappa* (log(NAct))^(1/2)./(conf*((maxiters)^(1/2)));
    eps = Kappa* (log(NAct))^(1/2)./(conf*((maxiters)^(1/2)));
    % eps = NPlayers*s*Kappa* (log(NAct))^(1/2)./(conf*((maxiters)^(3/4)));
elseif switch_eps == 0 || switch_eps == 2
    eps = (Kappa * conf);
elseif switch_eps == 3
    eps = (Kappa * conf * (log(NAct))^(1/2));
elseif switch_eps == 4
    eps = (Kappa/((maxiters)^(1/2)) * (4*sqrt(NAct*log(NAct))+2*(sqrt(NAct/log(NAct)))*log(2/conf)));
    %eps = (marg_distrib).^(1/2)./((maxiters)^(1/2)) .* (4*sqrt(NAct*log(NAct))+2*(sqrt(NAct/log(NAct)))*log(2/conf));
elseif switch_eps == 5
    eps = (NAct-1)./(Kappa*conf*maxiters);
end



end