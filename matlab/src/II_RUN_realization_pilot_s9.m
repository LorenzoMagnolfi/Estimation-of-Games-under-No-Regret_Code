%% II_RUN_realization_pilot_s9
%
%  Realization-conditional identification, FINER COST GRID test.
%
%  Mirror of II_RUN_realization_pilot but with s = 9 instead of s = 5.
%  Cost grid (linspace(0,6,9)): [0, 0.75, 1.5, 2.25, 3.0, 3.75, 4.5, 5.25, 6.0].
%  Truth at index 5 (cost = 3.0).  Same action grid, same N grid.
%
%  Diagnostic: does the identified set proportionally tighten with the
%  cost grid (--> grid coarseness was the floor in s=5), or does it stay
%  similar in cost-space range (--> LP flexibility at unrealized types
%  is the binding issue)?

clear all; clc; close all;

%% Setup
paths = df_repo_paths();
rng(20260429 + 9);  % distinct seed from s=5 run

NPlayers = 2;
alpha = -(1/3);
actions_vec = [4; 5; 6; 7; 8];
mu = 3 * ones(NPlayers, 1);
sigma2 = 1 * eye(NPlayers);
s = 9;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
fprintf('\n=== Realization-conditional pilot (s=%d) ===\n', s);
fprintf('Cost grid (player 1): [%s]\n', sprintf('%.3f ', cfg.type_space{1,1}));

%% True realization: median index
t_star_idx = [(s+1)/2, (s+1)/2];   % index 5 for s=9 -> cost 3.0
t_star_costs = [cfg.type_space{1,1}(t_star_idx(1)), ...
                cfg.type_space{2,1}(t_star_idx(2))];
fprintf('Truth: (t1*, t2*) = (%d, %d), costs (%.3f, %.3f)\n', ...
        t_star_idx(1), t_star_idx(2), t_star_costs(1), t_star_costs(2));

%% Sample sizes (mirrors s=5 pilot)
maxiters_values = [500000, 1000000, 2000000, 4000000];
n_iters = numel(maxiters_values);
alpha_set = 0.05;
switch_eps = 10;

%% Constants
s2 = s^2;
NA = cfg.NActPr;
a_dim = cfg.NAct;
dim_u = NA - 1;
dineq_obed = NPlayers * s * a_dim;

bmarg_uniform = ones(s2, 1) / s2;
marg_per_player = ones(s, 1) / s;

% Kronecker index map: (i, j) -> realized_idx in 1..s2
kron_idx = @(i, j) (j - 1) * s + i;

%% Loop
results = struct();
results.maxiters_values = maxiters_values;
results.t_star_idx = t_star_idx;
results.t_star_costs = t_star_costs;
results.s = s;
results.cost_grid = cfg.type_space{1,1};
results.VV = nan(n_iters, s, s);
results.distY_all = cell(n_iters, 1);
results.run_log = cell(n_iters, 1);

for n_idx = 1:n_iters
    M = maxiters_values(n_idx);
    fprintf('\n--- N = %s ---\n', commas(M));

    t_learn = tic;
    distY = df.sim.learn_fixed(cfg, M, t_star_idx);
    t_learn_val = toc(t_learn);
    fprintf('  Learn (fixed): %.1fs.  m_N support: %d/%d action profiles.\n', ...
            t_learn_val, sum(distY > 0), NA);
    results.distY_all{n_idx} = distY;

    eps_vec = df.solvers.compute_epsilon(cfg, M, alpha_set, switch_eps);
    eps_fin = repmat(sqrt(marg_per_player)', 1, NPlayers * a_dim) .* ...
              repmat(eps_vec, 1, NPlayers * a_dim);
    pi_realized = 1 / s2;

    t_lp = tic;
    n_feasible = 0;
    for i = 1:s
        for j = 1:s
            realized_idx = kron_idx(i, j);
            cstr = df.solvers.build_constraints_realization( ...
                cfg.type_space, cfg.action_space, cfg.Pi, realized_idx);
            c = [zeros(1, dim_u), ...
                 pi_realized * distY', ...
                 bmarg_uniform', ...
                 1, ...
                 eps_fin]';
            [optval, ~] = df.solvers.solve_socp_cvx(cstr, c, 'sedumi', 'default');
            results.VV(n_idx, i, j) = optval;
            if optval <= 1e-6
                n_feasible = n_feasible + 1;
            end
        end
    end
    t_lp_val = toc(t_lp);
    fprintf('  LP grid (%dx%d = %d candidates): %.1fs.  Feasible: %d/%d.\n', ...
            s, s, s*s, t_lp_val, n_feasible, s*s);
    results.run_log{n_idx} = struct('t_learn', t_learn_val, 't_lp', t_lp_val, ...
                                    'n_feasible', n_feasible);
end

%% Save
out_dir = fullfile(paths.matlab_root, 'output', 'realization_pilot');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'results_realization_R1_s9.mat'), '-struct', 'results');
fprintf('\nSaved results to %s\n', fullfile(out_dir, 'results_realization_R1_s9.mat'));

%% 4-panel heatmap (cost-axis labeling, no per-cell text — too small at s=9)
fig = figure('Color', 'w', 'Position', [80 80 1200 1100]);
cost_grid = cfg.type_space{1,1};

for n_idx = 1:n_iters
    subplot(2, 2, n_idx);
    VV = squeeze(results.VV(n_idx, :, :));
    feasible = VV <= 1e-6;

    imagesc(double(feasible));
    colormap(gca, [0.92 0.92 0.92; 0.20 0.55 0.85]);
    caxis([0 1]);
    set(gca, 'YDir', 'normal');
    axis equal tight;
    set(gca, 'XTick', 1:s, 'YTick', 1:s, ...
             'XTickLabel', arrayfun(@(c) sprintf('%.2f', c), cost_grid, 'UniformOutput', false), ...
             'YTickLabel', arrayfun(@(c) sprintf('%.2f', c), cost_grid, 'UniformOutput', false));
    xtickangle(45);
    xlabel('$c_2$ candidate', 'Interpreter', 'latex', 'FontSize', 13);
    ylabel('$c_1$ candidate', 'Interpreter', 'latex', 'FontSize', 13);

    hold on;
    plot(t_star_idx(2), t_star_idx(1), 'p', 'MarkerSize', 22, ...
         'MarkerEdgeColor', [0.6 0 0], 'MarkerFaceColor', [0.95 0.85 0.20], ...
         'LineWidth', 1.4);
    hold off;

    title(sprintf('$N = %s$ \\quad ($%d$/$%d$ feasible)', ...
                  commas(maxiters_values(n_idx)), ...
                  results.run_log{n_idx}.n_feasible, s*s), ...
          'Interpreter', 'latex', 'FontSize', 12);
    set(gca, 'FontSize', 9);
end

sgtitle({'Realization-conditional identified set: $\Lambda_N$ on $T_1\times T_2$ (s=9, finer grid)', ...
         sprintf('Truth $(c_1^*,c_2^*)=(%.2f,%.2f)$;  $|A|=5$; uniform prior; R1 radius', ...
                 t_star_costs(1), t_star_costs(2))}, ...
        'Interpreter', 'latex', 'FontSize', 14);

fig_dir = fullfile(paths.matlab_root, 'output', 'figures', 'realization_pilot');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
saveas(fig, fullfile(fig_dir, 'heatmap_realization_R1_s9.png'));
saveas(fig, fullfile(fig_dir, 'heatmap_realization_R1_s9.pdf'));
fprintf('Saved figure to %s\n', fullfile(fig_dir, 'heatmap_realization_R1_s9.png'));

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
