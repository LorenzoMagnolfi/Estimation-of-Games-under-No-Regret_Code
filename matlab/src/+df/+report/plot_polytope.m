function fig = plot_polytope(VP, save_path, opts)
% DF.REPORT.PLOT_POLYTOPE  Plot BCE polytope on 3-simplex using convhull.
%
%   fig = df.report.plot_polytope(VP, save_path)
%   fig = df.report.plot_polytope(VP, save_path, opts)
%
%   Native MATLAB replacement for drawBCCE (no MPT3/Polyhedron dependency).
%   Uses convhull + trisurf to render the polytope in 3D.
%
%   VP is N×4 (vertices on probability simplex over 4 action profiles).
%   Plots columns [2,3,1] as (x,y,z) to match the original drawBCCE convention.
%
%   opts fields:
%     .alpha       — face transparency (default: 0.3)
%     .color       — polytope face color (default: [1 0 0])
%     .show_simplex — draw simplex wireframe (default: true)
%     .view_angle  — [az, el] (default: [105, 35])
%     .font_size   — axis label font size (default: 24)

if nargin < 3, opts = struct(); end
if ~isfield(opts, 'alpha'),        opts.alpha = 0.3; end
if ~isfield(opts, 'color'),        opts.color = [1 0 0]; end
if ~isfield(opts, 'show_simplex'), opts.show_simplex = true; end
if ~isfield(opts, 'view_angle'),   opts.view_angle = [105, 35]; end
if ~isfield(opts, 'font_size'),    opts.font_size = 24; end

% Reorder columns: [q(pL,pH), q(pH,pL), q(pH,pH)]
COORD = VP(:, [2 3 1]);

fig = figure('Name', 'BCE Polytope', 'NumberTitle', 'off', ...
    'GraphicsSmoothing', 'on', 'Color', 'white');
hold on;

% Draw simplex wireframe
if opts.show_simplex
    S = [1 0 0; 0 1 0; 0 0 1; 0 0 0];
    edges = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4];
    for e = 1:size(edges, 1)
        plot3(S(edges(e,:), 1), S(edges(e,:), 2), S(edges(e,:), 3), ...
            'Color', [0 0.6 0 0.15], 'LineWidth', 1);
    end
    % Simplex faces
    K_s = convhull(S(:,1), S(:,2), S(:,3));
    trisurf(K_s, S(:,1), S(:,2), S(:,3), ...
        'FaceColor', 'green', 'FaceAlpha', 0.08, 'EdgeColor', 'none');
end

% Draw polytope
if size(COORD, 1) >= 4
    K = convhull(COORD(:,1), COORD(:,2), COORD(:,3));
    trisurf(K, COORD(:,1), COORD(:,2), COORD(:,3), ...
        'FaceColor', opts.color, 'FaceAlpha', opts.alpha, ...
        'EdgeColor', [0.6 0 0], 'EdgeAlpha', 0.3, 'LineWidth', 0.5);
elseif size(COORD, 1) == 3
    % Degenerate: triangle
    fill3(COORD(:,1), COORD(:,2), COORD(:,3), opts.color, ...
        'FaceAlpha', opts.alpha);
end

% Labels (LaTeX)
xl = xlabel('$q\big(p_{\ell},p_h\big)$');
set(xl, 'Interpreter', 'latex', 'FontName', 'cmr10', 'FontSize', opts.font_size);

yl = ylabel('$q\big(p_h,p_{\ell} \big)$');
set(yl, 'Interpreter', 'latex', 'FontName', 'cmr10', 'FontSize', opts.font_size);

zl = zlabel('$q\big(p_h,p_h \big)$');
set(zl, 'Interpreter', 'latex', 'FontName', 'cmr10', 'FontSize', opts.font_size);

xticks([0 0.5 1]);
yticks([0 0.5 1]);
set(gca, 'TickLabelInterpreter', 'latex', 'FontName', 'cmr10', ...
    'FontSize', 21, 'Box', 'off');

view(opts.view_angle);
hold off;

% Save
if nargin >= 2 && ~isempty(save_path)
    saveas(fig, save_path, 'epsc');
    saveas(fig, [save_path '.png'], 'png');
    fprintf('  Saved: %s.eps + .png\n', save_path);
end

end
