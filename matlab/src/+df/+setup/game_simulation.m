function cfg = game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s)
% df.setup.game_simulation  Build config struct for simulation stages (I, II, IV).
%
%   cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s)
%
%   Parameters:
%     NPlayers    - number of players (typically 2)
%     alpha       - demand parameter (e.g. -1/3)
%     actions_vec - column vector of individual actions (e.g. [4;5;6;7;8])
%     mu          - NPlayers x 1 mean MC vector
%     sigma2      - NPlayers x NPlayers MC variance matrix
%     s           - number of types per player
%
%   Returns:
%     cfg - struct with all game-defining objects, replacing globals.

cfg = struct();
cfg.NPlayers = NPlayers;
cfg.alpha = alpha;
cfg.mu = mu;
cfg.sigma2 = sigma2;
cfg.s = s;

% Action space
action_space = cell(NPlayers, 1);
for ind = 1:NPlayers
    action_space{ind, 1} = actions_vec;
end
cfg.action_space = action_space;
cfg.AA = actions_vec;
cfg.A = df.util.allcomb(action_space{1,:}, action_space{2,:});
cfg.NAct = size(actions_vec, 1);
cfg.NActPr = size(cfg.A, 1);

% Type space
[cfg.type_space, cfg.marg_distrib] = marginal_cost_draws_v5(mu, sigma2, s);
cfg.Egrid = cfg.type_space{1,1}';

% Joint type profiles
s2 = s^2;
T_sorted = cfg.type_space{1,1};
for ind = 2:NPlayers
    T_sorted = [kron(cfg.type_space{ind,1}, ones(size(T_sorted,1),1)) ...
                kron(ones(size(cfg.type_space{ind,1},1),1), T_sorted)];
end

% Joint prior Psi from product of marginals
cfg.Psi = zeros(s2, 1);
for ii = 1:s
    for jj = 1:s
        kk = (ii-1)*s + jj;
        cfg.Psi(kk) = cfg.marg_distrib(ii) * cfg.marg_distrib(jj);
    end
end

% Type matrix (types x players)
cfg.tps = [cfg.type_space{1,:}, cfg.type_space{2,:}];

% Utility tensor Pi(NActPr, s, NPlayers) using logit choice probabilities
cp = zeros(cfg.NActPr, NPlayers);
cfg.Pi = zeros(cfg.NActPr, size(cfg.tps, 1), NPlayers);

for j = 1:NPlayers
    for aa = 1:cfg.NActPr
        cp(aa, j) = exp(alpha * cfg.A(aa,j)) / (1 + sum(exp(alpha * cfg.A(aa,:)), 2));
        for tt = 1:s
            cfg.Pi(aa, tt, j) = cp(aa, j) * (cfg.A(aa,j) - cfg.tps(tt,j));
        end
    end
end

% Learning style (default: regret matching)
cfg.learning_style = 'rm';

end
