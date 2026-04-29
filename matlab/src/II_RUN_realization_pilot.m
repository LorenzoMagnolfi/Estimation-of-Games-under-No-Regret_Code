%% II_RUN_realization_pilot
%
%  Realization-conditional identification pilot (App. SIM-RC).
%
%  Setup:
%   - Same canonical 2-seller pricing-game lab as the rate-landscape pilot.
%   - Cost grid: the 5 quantile points of TruncNormal(mu=3, sigma2=1) — same
%     implicit support as the parametric DGP.
%   - Prior pi: UNIFORM over type pairs (1/s^2 each).
%   - Realized type pair: (t1_star, t2_star) = (3, 3), median, persistent.
%   - DGP: df.sim.learn_fixed with these types FIXED throughout the trajectory.
%
%  Identification:
%   - Parameter of interest = (t1_star, t2_star) in T_1 x T_2.
%   - Candidate space: enumerate s^2 = 25 type-pair candidates.
%   - LP per candidate: slice-equality (data) + consistency + total mass + obedience.
%   - Feasible candidate (VV <= 1e-6) is in the identified set.
%
%  Sample sizes: N in {500k, 1M, 2M, 4M}.  One trajectory per N.
%  Output: 5x5 feasibility heatmap per N.

clear all; clc; close all;

%% Setup (matches II_MAIN_simul / rate_landscape_pilot)
paths = df_repo_paths();
rng(20260429);  % session date

NPlayers = 2;
alpha = -(1/3);
actions_vec = [4; 5; 6; 7; 8];
mu = 3 * ones(NPlayers, 1);
sigma2 = 1 * eye(NPlayers);
s = 5;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
fprintf('\n=== Realization-conditional pilot ===\n');
fprintf('Cost grid (player 1): [%s]\n', sprintf('%.3f ', cfg.type_space{1,1}));
fprintf('Cost grid (player 2): [%s]\n', sprintf('%.3f ', cfg.type_space{2,1}));

%% True realization: median type pair (symmetric baseline)
t_star_idx = [3, 3];
t_star_costs = [cfg.type_space{1,1}(t_star_idx(1)), ...
                cfg.type_space{2,1}(t_star_idx(2))];
fprintf('Truth: (t1*, t2*) = (%d, %d), costs (%.3f, %.3f)\n', ...
        t_star_idx(1), t_star_idx(2), t_star_costs(1), t_star_costs(2));

%% Sample sizes (mirrors rate_landscape_pilot)
maxiters_values = [500000, 1000000, 2000000, 4000000];
n_iters = numel(maxiters_values);
alpha_set = 0.05;
switch_eps = 10;  % Niccolo R1, full feedback

%% Constants for LP construction
s2 = s^2;
NA = cfg.NActPr;          % 25 joint action profiles
a_dim = cfg.NAct;         % 5 actions per player
dim_u = NA - 1;
dineq_obed = NPlayers * s * a_dim;  % 50

% Uniform prior over type pairs
bmarg_uniform = ones(s2, 1) / s2;
% Uniform per-player marginal (each type 1/s)
marg_per_player = ones(s, 1) / s;

% Kronecker index map: (t1_idx, t2_idx) -> realized_idx in 1..s2
%   T_sorted column 1 = kron(type_space{2}, ones(s,1)) (player 2 slow)
%   T_sorted column 2 = kron(ones(s,1), type_space{1}) (player 1 fast)
%   So row k = (j-1)*s + i  with i=t1_idx, j=t2_idx
kron_idx = @(i, j) (j - 1) * s + i;

%% Loop over sample sizes: (i) DGP, (ii) per-candidate LP
results = struct();
results.maxiters_values = maxiters_values;
results.t_star_idx = t_star_idx;
results.t_star_costs = t_star_costs;
results.s = s;
results.VV = nan(n_iters, s, s);   % (N_idx, t1_cand, t2_cand)
results.distY_all = cell(n_iters, 1);
results.run_log = cell(n_iters, 1);

for n_idx = 1:n_iters
    M = maxiters_values(n_idx);
    fprintf('\n--- N = %s ---\n', commas(M));

    %% (i) DGP: fixed-realization Hedge
    t_learn = tic;
    distY = df.sim.learn_fixed(cfg, M, t_star_idx);
    t_learn_val = toc(t_learn);
    fprintf('  Learn (fixed): %.1fs.  m_N support: %d/%d action profiles.\n', ...
            t_learn_val, sum(distY > 0), NA);
    results.distY_all{n_idx} = distY;

    %% (ii) Per-candidate LP
    % Epsilon: per-cell radius (R1, full feedback), 1xs vector
    eps_vec = df.solvers.compute_epsilon(cfg, M, alpha_set, switch_eps);

    % eps_fin: 1 x dineq_obed.  Per-cell sqrt(phi) factor with uniform marginal.
    eps_fin = repmat(sqrt(marg_per_player)', 1, NPlayers * a_dim) .* ...
              repmat(eps_vec, 1, NPlayers * a_dim);

    pi_realized = 1 / s2;

    t_lp = tic;
    n_feasible = 0;
    for i = 1:s
        for j = 1:s
            realized_idx = kron_idx(i, j);

            % Build constraints for this candidate
            cstr = df.solvers.build_constraints_realization( ...
                cfg.type_space, cfg.action_space, cfg.Pi, realized_idx);

            % Build objective vector c
            % Order matches dual form:
            %   [u (NA-1); slice_rhs (NA); consistency_rhs (s2); total_mass (1); eps_fin (dineq_obed)]
            c = [zeros(1, dim_u), ...
                 pi_realized * distY', ...
                 bmarg_uniform', ...
                 1, ...
                 eps_fin]';

            [optval, status] = df.solvers.solve_socp_cvx(cstr, c, 'sedumi', 'default');
            results.VV(n_idx, i, j) = optval;
            if optval <= 1e-6
                n_feasible = n_feasible + 1;
            end
        end
    end
    t_lp_val = toc(t_lp);
    fprintf('  LP grid (5x5 = 25 candidates): %.1fs.  Feasible: %d/%d.\n', ...
            t_lp_val, n_feasible, s*s);
    results.run_log{n_idx} = struct('t_learn', t_learn_val, 't_lp', t_lp_val, ...
                                    'n_feasible', n_feasible);
end

%% Save
out_dir = fullfile(paths.matlab_root, 'output', 'realization_pilot');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'results_realization_R1.mat'), '-struct', 'results');
fprintf('\nSaved results to %s\n', fullfile(out_dir, 'results_realization_R1.mat'));

%% 4-panel heatmap figure
fig = figure('Color', 'w', 'Position', [80 80 1200 1100]);
for n_idx = 1:n_iters
    subplot(2, 2, n_idx);
    VV = squeeze(results.VV(n_idx, :, :));
    feasible = VV <= 1e-6;

    % Heatmap of feasibility (1 = feasible, 0 = infeasible)
    imagesc(double(feasible));
    colormap(gca, [0.92 0.92 0.92; 0.20 0.55 0.85]);  % infeasible grey, feasible blue
    caxis([0 1]);
    set(gca, 'YDir', 'normal');
    axis equal tight;
    set(gca, 'XTick', 1:s, 'YTick', 1:s);
    xlabel('$t_2$ candidate', 'Interpreter', 'latex', 'FontSize', 13);
    ylabel('$t_1$ candidate', 'Interpreter', 'latex', 'FontSize', 13);

    % Star at the truth
    hold on;
    plot(t_star_idx(2), t_star_idx(1), 'p', 'MarkerSize', 22, ...
         'MarkerEdgeColor', [0.6 0 0], 'MarkerFaceColor', [0.95 0.85 0.20], ...
         'LineWidth', 1.4);

    % VV value annotations
    for i = 1:s
        for j = 1:s
            if feasible(i, j)
                txt = sprintf('%.1e', VV(i, j));
                col = [1 1 1];
            else
                txt = sprintf('%.1e', VV(i, j));
                col = [0.3 0.3 0.3];
            end
            text(j, i, txt, 'HorizontalAlignment', 'center', ...
                 'FontSize', 8, 'Color', col, 'Interpreter', 'tex');
        end
    end
    hold off;

    title(sprintf('$N = %s$ \\quad ($%d$/$%d$ feasible)', ...
                  commas(maxiters_values(n_idx)), ...
                  results.run_log{n_idx}.n_feasible, s*s), ...
          'Interpreter', 'latex', 'FontSize', 12);
    set(gca, 'FontSize', 10);
end

sgtitle({'Realization-conditional identified set: $\Lambda_N$ on $T_1\times T_2$', ...
         sprintf('Truth $(t_1^*,t_2^*)=(%d,%d)$;  $|A|=5$; uniform prior; R1 radius', ...
                 t_star_idx(1), t_star_idx(2))}, ...
        'Interpreter', 'latex', 'FontSize', 14);

fig_dir = fullfile(paths.matlab_root, 'output', 'figures', 'realization_pilot');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
saveas(fig, fullfile(fig_dir, 'heatmap_realization_R1.png'));
saveas(fig, fullfile(fig_dir, 'heatmap_realization_R1.pdf'));
fprintf('Saved figure to %s\n', fullfile(fig_dir, 'heatmap_realization_R1.png'));

%% Helper
function s = commas(n)
    str = sprintf('%d', n);
    L = length(str);
    s = '';
    for i = 1:L
        s = [s, str(i)];
        rem = L - i;
        if rem > 0 && mod(rem, 3) == 0
            s = [s, ','];
        end
    end
end
