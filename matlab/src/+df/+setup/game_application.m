function cfg = game_application(player_id, Dist_file, Prob_file, n_types)
% df.setup.game_application  Build config struct for application stage (III).
%
%   cfg = df.setup.game_application(player_id, Dist_file, Prob_file, n_types)
%
%   Parameters:
%     player_id  - which player to load data for (1 or 2)
%     Dist_file  - path to seller distribution xlsx
%     Prob_file  - path to sale probability xlsx
%     n_types    - number of types per player
%
%   Returns:
%     cfg - struct with all game-defining objects for the application,
%           replacing globals. Includes data-driven actions and sale
%           probabilities (not logit).

[distrib, actions, prob, maxiters] = get_player_data_5acts(player_id, 'median', Dist_file, Prob_file);

cfg = struct();
cfg.NPlayers = 2;       % always 2-player game in the application
cfg.s = n_types;
cfg.maxiters = maxiters;
cfg.distrib = distrib;
cfg.prob = prob;

% Action space bounds
P_l = actions(1);
P_h = actions(5);
diff_p = P_h - P_l;
mid = P_l + 0.5 * diff_p;

ub = P_h + 0.25 * diff_p;
lb = P_l - 3 * diff_p;

% Type space
cfg.type_space = cell(2, 1);
cfg.type_space{1,1} = linspace(lb, ub, n_types)';
cfg.type_space{2,1} = linspace(lb, ub, n_types)';

% Action space
NPlayer = 2;
action_space = cell(NPlayer, 1);
for ind = 1:NPlayer
    action_space{ind, 1} = actions';
end
cfg.action_space = action_space;
cfg.AA = action_space{1,1};
cfg.A = df.util.allcomb(action_space{1,:}, action_space{2,:});
cfg.NAct = size(action_space{1,1}, 1);
cfg.NActPr = size(cfg.A, 1);

% Marginal mean action distribution (opponent-marginal)
cfg.marg_mean = kron(ones(1, cfg.NAct), eye(cfg.NAct)) * distrib';

% Type matrix
cfg.tps = [cfg.type_space{1,:}, cfg.type_space{2,:}];

% Utility tensor using data-driven sale probabilities (not logit)
cp = zeros(cfg.NActPr, NPlayer);
cfg.Pi = zeros(cfg.NActPr, size(cfg.tps, 1), NPlayer);

for j = 1:NPlayer
    for aa = 1:cfg.NActPr
        cp(aa, j) = prob(aa);
        for tt = 1:n_types
            cfg.Pi(aa, tt, j) = cp(aa, j) * (cfg.A(aa,j) - cfg.tps(tt,j));
        end
    end
end

% MC parameters (for grid construction)
cfg.mu = mid;
cfg.sigma2 = 0.33 * diff_p * eye(NPlayer);

end
