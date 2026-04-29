function eps = compute_epsilon(cfg, maxiters, conf, switch_eps, kwargs)
% df.solvers.compute_epsilon  Unified epsilon dispatcher for all stages.
%
%   eps = df.solvers.compute_epsilon(cfg, maxiters, conf, switch_eps)
%   eps = df.solvers.compute_epsilon(cfg, maxiters, conf, switch_eps, kwargs)
%
%   Merges epsilon_switch.m (simulation, Stages I/II/IV) and
%   epsilon_switch_distrib.m (application, Stage III).
%
%   Parameters:
%     cfg        - game config struct (must have .Pi, .NAct, .NPlayers, .s)
%     maxiters   - number of learning iterations T
%     conf       - confidence level
%     switch_eps - epsilon formula selector (0-9)
%     kwargs     - (optional) struct with fields:
%       .mode      - 'simulation' (default) | 'application'
%       .marg_mean - required for application mode with switch_eps >= 6
%       .s_override - optional s override (application mode passes s explicitly)
%
%   NOTE on switch_eps==1 discrepancy: simulation mode uses
%       Kappa * sqrt(log(NAct)) / (conf * sqrt(T))
%   while application mode uses
%       s * Kappa * sqrt(log(NAct)) / (conf * sqrt(T))
%   This is intentional and matches the original epsilon_switch.m vs
%   epsilon_switch_distrib.m behavior.

if nargin < 5
    kwargs = struct();
end

mode = 'simulation';
if isfield(kwargs, 'mode')
    mode = kwargs.mode;
end

Pi = cfg.Pi;
NAct = cfg.NAct;

% Kappa computation (shared across both modes)
if switch_eps <= 1 || switch_eps == 3 || switch_eps == 4 || switch_eps == 10 || switch_eps == 11
    Kappa = max(Pi(:,:,1)) - min(Pi(:,:,1));
elseif switch_eps == 2
    Kappa = max(abs(Pi(1:2,:,1) - Pi(3:4,:,1)));
elseif switch_eps == 5
    for jj = 1:size(Pi, 2)
        Kappa(jj) = min(pdist(Pi(:,jj,1), 'euclidean'));
    end
elseif switch_eps == 6 || switch_eps == 7
    Kappa = max(Pi(:,:,1)) - min(Pi(:,:,1));
    marg_mean = kwargs.marg_mean;
    for jj = 1:size(Pi, 2)
        JL = kron(eye(NAct), marg_mean') * Pi(:,jj,1);
        Ddeltak = abs(max(JL) - JL);
        TType_spec_LargestDev(jj) = Kappa(jj) * (sum(Ddeltak(Ddeltak > 0).^(-1)));
        TType_spec_LargestDev2(jj) = Kappa(jj) * (sum(Ddeltak(Ddeltak > 0.1*Kappa(jj)).^(-1)));
    end
elseif switch_eps == 8 || switch_eps == 9
    marg_mean = kwargs.marg_mean;
    for jj = 1:size(Pi, 2)
        JL = kron(eye(NAct), marg_mean') * Pi(:,jj,1);
        Ddeltak = abs(max(JL) - JL);
        bar_Deltak = max(Ddeltak);
        TType_spec_LargestDev2(jj) = bar_Deltak * (sum(Ddeltak(Ddeltak > 0.1*bar_Deltak).^(-1)));
    end
end

% Epsilon computation
if strcmp(mode, 'simulation')
    % Original epsilon_switch.m formulas
    if switch_eps == 1
        eps = Kappa .* sqrt(log(NAct)) ./ (conf * sqrt(maxiters));
    elseif switch_eps == 0 || switch_eps == 2
        eps = Kappa * conf;
    elseif switch_eps == 3
        eps = Kappa * conf * sqrt(log(NAct));
    elseif switch_eps == 4
        eps = Kappa / sqrt(maxiters) * (4*sqrt(NAct*log(NAct)) + 2*sqrt(NAct/log(NAct))*log(2/conf));
    elseif switch_eps == 5
        eps = (NAct - 1) ./ (Kappa * conf * maxiters);
    elseif switch_eps == 10
        % Niccolo R1: adversarial environment + full feedback (Hedge corollary)
        % Per-cell radius (sqrt(phi) factored out at LP layer):
        %   K * sqrt(2*ln|A|) / (alpha * sqrt(N))
        % vs switch_eps==1: factor of sqrt(2) wider.
        % The K*ln|A|/(alpha*N) lower-order correction is omitted (negligible
        % at our N and cannot be threaded through the per-cell sqrt(phi) factor).
        eps = Kappa .* sqrt(2 * log(NAct)) ./ (conf * sqrt(maxiters));
    elseif switch_eps == 11
        % Niccolo R2: adversarial environment + bandit feedback (Exp3 corollary)
        % Per-cell radius (sqrt(phi) factored out at LP layer):
        %   2*sqrt(e-1) * K * sqrt(|A|*ln|A|) / (alpha * sqrt(N))
        % vs switch_eps==1: factor of 2*sqrt(e-1)*sqrt(|A|) wider.
        eps = 2 * sqrt(exp(1) - 1) * Kappa .* sqrt(NAct * log(NAct)) ./ (conf * sqrt(maxiters));
    end
else
    % Original epsilon_switch_distrib.m formulas
    if isfield(kwargs, 's_override')
        s_val = kwargs.s_override;
    else
        s_val = cfg.s;
    end

    if switch_eps == 1
        % NOTE: application mode has s* multiplier; simulation does not
        eps = 1 * s_val * (Kappa .* sqrt(log(NAct))) ./ (conf * sqrt(maxiters));
    elseif switch_eps == 0 || switch_eps == 2
        eps = Kappa * conf;
    elseif switch_eps == 3
        eps = Kappa * conf * sqrt(log(NAct));
    elseif switch_eps == 4
        eps = Kappa / sqrt(maxiters) * (4*sqrt(NAct*log(NAct)) + 2*sqrt(NAct/log(NAct))*log(2/conf));
    elseif switch_eps == 5 || switch_eps == 6
        eps = TType_spec_LargestDev ./ (conf * maxiters);
    elseif switch_eps == 7 || switch_eps == 8
        eps = s_val .* TType_spec_LargestDev2 ./ (conf * maxiters);
    elseif switch_eps == 9
        eps = TType_spec_LargestDev2 ./ ((1 - (1-conf)^(1/s_val)) * maxiters);
    elseif switch_eps == 10
        % Niccolo R1 (application analogue): preserves switch_eps==1's s_val multiplier.
        eps = s_val * Kappa .* sqrt(2 * log(NAct)) ./ (conf * sqrt(maxiters));
    elseif switch_eps == 11
        % Niccolo R2 (application analogue): preserves switch_eps==1's s_val multiplier.
        eps = s_val * 2 * sqrt(exp(1) - 1) * Kappa .* sqrt(NAct * log(NAct)) ./ (conf * sqrt(maxiters));
    end
end

end
