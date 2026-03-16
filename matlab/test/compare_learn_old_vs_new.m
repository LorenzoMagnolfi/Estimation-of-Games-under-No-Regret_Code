%% compare_learn_old_vs_new.m — Verify df.sim.learn matches original learn_mod
%
% Runs the OLD learn_mod code (inlined) and the NEW df.sim.learn side by
% side with the same RNG seed, comparing all outputs to machine precision.

this_file = mfilename('fullpath');
test_dir = fileparts(this_file);
matlab_root = fileparts(test_dir);
src_dir = fullfile(matlab_root, 'src');
addpath(src_dir);

fprintf('=== learn_mod OLD vs df.sim.learn NEW comparison ===\n\n');

%% Setup (identical to fixture runner Stage II)
NPlayers = 2;
alpha = -(1/3);
actions_vec = [4;5;6;7;8];
mu = 3*ones(NPlayers,1);
sigma2 = 1*eye(NPlayers);
s = 5;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
cfg.learning_style = 'rm';

N = 1;
M = 5000;
M_obs = 5000;
numdst_t = 1;
numdst_t_obs = 1;
Nobs_Pl1 = 1000;
Nobs_Pl2 = 2000;

%% Run OLD (inlined original learn_mod code)
fprintf('Running OLD learn_mod (inlined)...');
rng(12345);
t_old = tic;

A = cfg.A; AA = cfg.AA;
type_space = cfg.type_space;

% -- BEGIN original learn_mod code --
U_n_old = zeros(s, NPlayers);
Avg_cf_util_old = zeros(size(AA,1), s, NPlayers);
type_counts_old = zeros(s, NPlayers);
distY_time_old = zeros(size(A,1), numdst_t);
distY_time_obs_old = zeros(size(A,1), numdst_t_obs);
action_counts_old = zeros(size(A,1), 1);
final_regret_old = zeros(s, NPlayers);
Pl1_EmpRegr_old = zeros(s, 1);
Pl2_EmpRegr_old = zeros(s, 1);

for t = 1:(N+M)
    mc_draw = marginal_cost_draws_v4_new(cfg, type_space, 1);
    type_indices_old = zeros(1, NPlayers);
    for j = 1:NPlayers
        type_indices_old(j) = find(type_space{j,1} == mc_draw(j));
    end

    if t <= N
        action_indices = randi(size(A, 1), 1, 1);
        actions = A(action_indices, :);
    else
        actions = zeros(1, NPlayers);
        for j = 1:NPlayers
            [actions(j), ~] = regret_matching_mod(AA, U_n_old(:,j), Avg_cf_util_old(:,:,j), type_indices_old(j));
        end
    end

    for j = 1:NPlayers
        type_j = type_indices_old(j);
        type_counts_old(type_j, j) = type_counts_old(type_j, j) + 1;
        utility = choice_prob(actions(j), actions, j, alpha) * (actions(j) - mc_draw(j));
        for tt = 1:s
            if tt == type_j
                U_n_old(tt, j) = ((t-1) * U_n_old(tt, j) + utility) / t;
                for a = 1:size(AA,1)
                    cf_utility = choice_prob(AA(a), actions, j, alpha) * (AA(a) - mc_draw(j));
                    Avg_cf_util_old(a, tt, j) = ((t-1) * Avg_cf_util_old(a, tt, j) + cf_utility) / t;
                end
            else
                U_n_old(tt, j) = ((t-1) * U_n_old(tt, j)) / t;
                for a = 1:size(AA,1)
                    Avg_cf_util_old(a, tt, j) = ((t-1) * Avg_cf_util_old(a, tt, j)) / t;
                end
            end
        end
    end

    action_index = find(all(A == actions, 2));
    action_counts_old(action_index) = action_counts_old(action_index) + 1;

    if ismember(t, round(M * (1:numdst_t)/numdst_t))
        distY_time_old(:, find(round(M * (1:numdst_t)/numdst_t) == t)) = action_counts_old / t;
    end

    if t > N+M-M_obs && ismember(t-(N+M-M_obs), round(M_obs * (1:numdst_t_obs)/numdst_t_obs))
        distY_time_obs_old(:, find(round(M_obs * (1:numdst_t_obs)/numdst_t_obs) == t-(N+M-M_obs))) = ...
            action_counts_old / t;
    end

    if t == N+M
        for j = 1:NPlayers
            for type_j = 1:s
                [~, final_regret_old(type_j,j)] = regret_matching_mod(AA, U_n_old(:,j), Avg_cf_util_old(:,:,j), type_j);
            end
        end
    elseif t == N+Nobs_Pl1
        for type_j = 1:s
            [~, Pl1_EmpRegr_old(type_j)] = regret_matching_mod(AA, U_n_old(:,1), Avg_cf_util_old(:,:,1), type_j);
        end
    elseif t == N+Nobs_Pl2
        for type_j = 1:s
            [~, Pl2_EmpRegr_old(type_j)] = regret_matching_mod(AA, U_n_old(:,2), Avg_cf_util_old(:,:,2), type_j);
        end
    end
end
% -- END original learn_mod code --
t_old_elapsed = toc(t_old);
fprintf(' %.2fs\n', t_old_elapsed);

%% Run NEW df.sim.learn
fprintf('Running NEW df.sim.learn...');
rng(12345);
t_new = tic;
[distY_time_new, distY_time_obs_new, final_regret_new, Pl1_EmpRegr_new, Pl2_EmpRegr_new] = ...
    df.sim.learn(cfg, N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2);
t_new_elapsed = toc(t_new);
fprintf(' %.2fs\n', t_new_elapsed);

%% Compare
fprintf('\n=== Comparison ===\n');
all_pass = true;

fields = {'distY_time', 'distY_time_obs', 'final_regret', 'Pl1_EmpRegr', 'Pl2_EmpRegr'};
old_vals = {distY_time_old, distY_time_obs_old, final_regret_old, Pl1_EmpRegr_old, Pl2_EmpRegr_old};
new_vals = {distY_time_new, distY_time_obs_new, final_regret_new, Pl1_EmpRegr_new, Pl2_EmpRegr_new};

for k = 1:numel(fields)
    d = max(abs(old_vals{k}(:) - new_vals{k}(:)));
    if d < 1e-12
        status = 'PASS';
    else
        status = 'FAIL';
        all_pass = false;
    end
    fprintf('  [%s] %-20s max_diff=%.2e\n', status, fields{k}, d);
end

fprintf('\nTiming: OLD=%.2fs  NEW=%.2fs  speedup=%.1fx\n', ...
    t_old_elapsed, t_new_elapsed, t_old_elapsed/t_new_elapsed);

if all_pass
    fprintf('\n=== ALL PASS: df.sim.learn is numerically identical to learn_mod ===\n');
else
    fprintf('\n=== SOME FAILURES: outputs differ ===\n');
end
