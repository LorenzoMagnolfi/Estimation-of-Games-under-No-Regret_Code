function eps = epsilon_switch_distrib(maxiters,conf,switch_eps,marg_mean,s)
 
% SwitchEps = 1 --- eps with new conv rates & INTERS BOUNDS
% SwitchEps = 0 --- fixed eps with new largest dev
% SwitchEps = 2 --- fixed eps with OLD largest dev ONLY WORKS FOR 2
% ACTIONS!
% SwitchEps = 3 --- fixed eps with NEW DEF Feb 2024
% SwitchEps = 4 --- EXP3 bounds
% SwitchEps = 5 --- Stochastic bandit w/ full monitoring, conservative
% approach
% SwitchEps = 6 --- Stochastic bandit w/ full monitoring, averaging over
% SwitchEps = 7 --- Stochastic bandit w/ full monitoring, averaging over +
% intersec bds
% SwitchEps = 8 --- Stochastic bandit w/ full monitoring, averaging over +
% intersec bds; UPDATED w/ tighter bds
% SwitchEps = 9 --- Stochastic bandits w/ full monitoring + indep,
% intersect bds
% payoffs

global Pi NAct NPlayers
if switch_eps <= 1 || switch_eps==3 || switch_eps==4
    Kappa = max(Pi(:,:,1))-(min(Pi(:,:,1))); % will be used to multiply eps!
elseif switch_eps == 2
    Kappa = max(abs(Pi(1:2,:,1)-Pi(3:4,:,1))); 
elseif switch_eps == 5
    for jj = 1:size(Pi,2)
    Kappa(jj) = min(pdist(Pi(:,jj,1), 'euclidean'));
    end
elseif switch_eps == 6 || switch_eps == 7
    Kappa = max(Pi(:,:,1))-(min(Pi(:,:,1))); % will be used to multiply eps!
    for jj = 1:size(Pi,2)
    JL = kron(eye(NAct),marg_mean')*Pi(:,jj,1);
    Ddeltak = abs(max(JL) - JL);
    TType_spec_LargestDev(jj) = Kappa(jj)*(sum(Ddeltak(Ddeltak>0).^(-1)));
    TType_spec_LargestDev2(jj) = Kappa(jj)*(sum(Ddeltak(Ddeltak>0.1*Kappa(jj)).^(-1)));
    end
elseif switch_eps == 8 || switch_eps == 9
    for jj = 1:size(Pi,2)
    JL = kron(eye(NAct),marg_mean')*Pi(:,jj,1);
    Ddeltak = abs(max(JL) - JL);
    bar_Deltak = max(Ddeltak);
    TType_spec_LargestDev2(jj) = bar_Deltak*(sum(Ddeltak(Ddeltak>0.1*bar_Deltak).^(-1)));
%   TType_spec_LargestDev2(jj) = bar_Deltak*(sum(Ddeltak(Ddeltak>0).^(-1)));
%   TType_spec_LargestDev2(jj) = bar_Deltak*(sum(Ddeltak(Ddeltak>0.03*bar_Deltak).^(-1)));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% EPS linked to rate of convergence!
if switch_eps == 1
    %eps = NPlayers*s*(Kappa * (log(NAct))^(1/2))./(conf*((maxiters)^(1/2)));
    eps = 1*s*(Kappa * (log(NAct))^(1/2))./(conf*((maxiters)^(1/2)));
elseif switch_eps == 0 || switch_eps == 2
    eps = (Kappa * conf);
elseif switch_eps == 3
    eps = (Kappa * conf * (log(NAct))^(1/2));
elseif switch_eps == 4
    eps = (Kappa/((maxiters)^(1/2)) * (4*sqrt(NAct*log(NAct))+2*(sqrt(NAct/log(NAct)))*log(2/conf)));
    %eps = (marg_distrib).^(1/2)./((maxiters)^(1/2)) .* (4*sqrt(NAct*log(NAct))+2*(sqrt(NAct/log(NAct)))*log(2/conf));
elseif switch_eps == 5 || switch_eps == 6
    eps = TType_spec_LargestDev./(conf*maxiters);
elseif switch_eps == 7 || switch_eps == 8
    %eps = s.*TType_spec_LargestDev./(conf*maxiters);
    eps = s.*TType_spec_LargestDev2./(conf*maxiters);
elseif switch_eps == 9
    eps = TType_spec_LargestDev2./((1-(1-conf)^(1/s))*maxiters);
end



end