function fig = plot_convergence(cfg, save_path, numdists, maxiters, N, actions, VP)
% DF.REPORT.PLOT_CONVERGENCE  Plot learning convergence on 3-simplex.
%
%   Native MATLAB replacement for drawConvergence (no MPT3 dependency).
%   Shows scatter of empirical action distributions converging toward
%   the BCE polytope, rendered via convhull/trisurf.
%
%   Inputs:
%     cfg       — config struct (must have .A)
%     save_path — base path for .epsc/.png output
%     numdists  — number of sample points along learning path (default: 1000)
%     maxiters  — total learning iterations
%     N         — initial observations count
%     actions   — (N+maxiters) × NPlayers action history
%     VP        — polytope vertices (from solve_polytope_lp)

A = cfg.A;

fig = figure('Name', 'Learning Convergence', 'NumberTitle', 'off', ...
    'GraphicsSmoothing', 'on', 'Color', 'white');
hold on;

% Draw simplex wireframe
S = [1 0 0; 0 1 0; 0 0 1; 0 0 0];
edges = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4];
for e = 1:size(edges, 1)
    plot3(S(edges(e,:), 1), S(edges(e,:), 2), S(edges(e,:), 3), ...
        'Color', [0 0.6 0 0.15], 'LineWidth', 1);
end
K_s = convhull(S(:,1), S(:,2), S(:,3));
trisurf(K_s, S(:,1), S(:,2), S(:,3), ...
    'FaceColor', 'green', 'FaceAlpha', 0.08, 'EdgeColor', 'none');

% Compute empirical distributions at sample points
VP_2 = zeros(numdists, size(A, 1));
ind1 = 1;
for obs = round(maxiters * (1:numdists) / numdists)
    for ind2 = 1:size(A, 1)
        VP_2(ind1, ind2) = sum(prod(actions(1:(N+obs), :) == A(ind2, :), 2)) / (N+obs);
    end
    ind1 = ind1 + 1;
end

% Scatter: columns [2,3,4] = [q(pL,pH), q(pH,pL), q(pH,pH)]
COORD_scatter = VP_2(:, [2 3 4]);
s_size = 10 * ones(numdists, 1);
c_color = linspace(0, 1, numdists)';
scatter3(COORD_scatter(:,1), COORD_scatter(:,2), COORD_scatter(:,3), s_size, c_color);

% Draw BCE polytope
COORD_poly = VP(:, [2 3 1]);
if size(COORD_poly, 1) >= 4
    K = convhull(COORD_poly(:,1), COORD_poly(:,2), COORD_poly(:,3));
    trisurf(K, COORD_poly(:,1), COORD_poly(:,2), COORD_poly(:,3), ...
        'FaceColor', [1 0 0], 'FaceAlpha', 0.3, ...
        'EdgeColor', [0.6 0 0], 'EdgeAlpha', 0.3, 'LineWidth', 0.5);
end

% Labels
xl = xlabel('$q\big(p_{\ell},p_h\big)$');
set(xl, 'Interpreter', 'latex', 'FontName', 'cmr10', 'FontSize', 16);

yl = ylabel('$q\big(p_h,p_{\ell} \big)$');
set(yl, 'Interpreter', 'latex', 'FontName', 'cmr10', 'FontSize', 16);

zl = zlabel('$q\big(p_h,p_h \big)$');
set(zl, 'Interpreter', 'latex', 'FontName', 'cmr10', 'FontSize', 16);

xticks([0 0.25 0.5 0.75 1]);
yticks([0 0.25 0.5 0.75 1]);
zticks([0 0.25 0.5 0.75 1]);
set(gca, 'TickLabelInterpreter', 'latex', 'FontName', 'cmr10', ...
    'FontSize', 15, 'Box', 'off');

view(105, 35);
hold off;

% Save
if ~isempty(save_path)
    saveas(fig, save_path, 'epsc');
    saveas(fig, [save_path '.png'], 'png');
    fprintf('  Saved: %s.eps + .png\n', save_path);
end

end
