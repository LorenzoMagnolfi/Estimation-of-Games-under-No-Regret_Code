%% III_DIAGNOSE_mN
%
%  Diagnostic on the empirical action distribution m_N per seller in the
%  application data.  Goal: figure out *why* m_N is "uniform 25/25" for
%  Sellers 1 and 2 in our earlier trial — is it actually uniform, or is
%  it concentrated with non-zero tails, or something else?
%
%  Report per seller: support size, max-mass profile, entropy, Herfindahl,
%  cumulative mass on top-K profiles.

% Allow caller to set aggregation_mode = 'avg' (default) or 'minprice'.
% Caller sets it in workspace before calling this script; we leave it alone.
if ~exist('aggregation_mode', 'var')
    aggregation_mode = 'avg';
end
clc; close all;

paths = df_repo_paths();
if strcmp(aggregation_mode, 'minprice')
    Dist_file = fullfile(paths.data, 'SellerDistribution_15_sellers_res1_minprice.xlsx');
    fig_suffix = '_minprice';
    fprintf('\n[Aggregation mode: MIN-OF-OTHERS competing price]\n');
else
    aggregation_mode = 'avg';
    Dist_file = fullfile(paths.data, 'SellerDistribution_15_sellers_res1.xlsx');
    fig_suffix = '';
    fprintf('\n[Aggregation mode: AVG-OF-OTHERS competing price]\n');
end

n_sellers = 15;
n_profiles = 25;

fprintf('\n=== m_N concentration diagnostic ===\n');
fprintf('Per seller: support, max mass, top-3 cumulative, entropy, HHI\n\n');
fprintf('%-7s | %-7s | %-9s | %-9s | %-7s | %-7s\n', ...
    'Seller', 'support', 'max mass', 'top-3 sum', 'entropy', 'HHI');
fprintf('%s\n', repmat('-', 1, 70));

mN_all = nan(n_sellers, n_profiles);

for sid = 1:n_sellers
    sheet_name = sprintf('Seller_%d', sid);
    try
        T = readmatrix(Dist_file, 'Sheet', sheet_name, 'NumHeaderLines', 1);
    catch
        fprintf('Seller %d: failed to read sheet "%s"\n', sid, sheet_name);
        continue
    end
    % Final time-averaged distribution = last row
    mN = T(end, :);
    mN = mN(:)';                     % row vector
    mN(isnan(mN)) = 0;
    mN = mN / sum(mN);              % renormalize defensively

    mN_all(sid, :) = mN;

    support = sum(mN > 1e-6);
    max_mass = max(mN);
    [sorted_mN, ord] = sort(mN, 'descend');
    top3 = sum(sorted_mN(1:3));
    entropy_val = -sum(mN(mN > 0) .* log(mN(mN > 0)));     % nats
    HHI = sum(mN.^2);                                       % Herfindahl

    fprintf('%-7d | %3d/%-3d | %9.4f | %9.4f | %7.4f | %7.4f\n', ...
        sid, support, n_profiles, max_mass, top3, entropy_val, HHI);
end

fprintf('\nReference: uniform on 25 profiles -> max=0.04, top-3=0.12, entropy=%.4f, HHI=%.4f\n', ...
    log(25), 1/25);

%% Top-3 dominant profiles per seller
fprintf('\n--- Top-3 profiles per seller (profile-idx: mass) ---\n');
for sid = 1:n_sellers
    if all(isnan(mN_all(sid, :))), continue; end
    mN = mN_all(sid, :);
    [sorted_mN, ord] = sort(mN, 'descend');
    fprintf('Seller %2d:', sid);
    for k = 1:3
        fprintf('  [%d: %.3f]', ord(k), sorted_mN(k));
    end
    % Translate profile index to (own_bin, comp_bin)
    % Profile is 1..25 numbered with comp slow, self fast (or vice versa)
    % From generate_matlab_data.do, profile = (self-1)*5 + comp, 0-indexed
    %   i_00 = (self=1, comp=1) -> col 1
    %   i_01 = (self=1, comp=2) -> col 2
    %   ...
    %   i_44 = (self=5, comp=5) -> col 25
    % So col = (self - 1) * 5 + comp
    fprintf('  (self, comp) bins: ');
    for k = 1:3
        col = ord(k);
        self_bin = floor((col - 1) / 5) + 1;
        comp_bin = mod(col - 1, 5) + 1;
        fprintf('(%d,%d) ', self_bin, comp_bin);
    end
    fprintf('\n');
end

%% Visual: heatmap of m_N for each seller (5x5 grid: self x comp)
fig = figure('Color', 'w', 'Position', [50 50 1500 900]);
for sid = 1:n_sellers
    if all(isnan(mN_all(sid, :))), continue; end
    subplot(3, 5, sid);
    mN = mN_all(sid, :);
    M = reshape(mN, 5, 5)';            % rows = self_bin (1..5), cols = comp_bin (1..5)
    % Wait: profile col = (self-1)*5 + comp.  mN(1) is (self=1, comp=1).
    % reshape(mN, 5, 5) groups by first dim varying fastest -> rows correspond
    % to comp, cols to self.  Want rows=self, cols=comp -> transpose.
    imagesc(M);
    colormap(gca, 'hot');
    colorbar;
    set(gca, 'YDir', 'normal');
    axis image;
    set(gca, 'XTick', 1:5, 'YTick', 1:5);
    xlabel('comp bin');
    ylabel('self bin');
    title(sprintf('Seller %d', sid), 'FontSize', 10);
    set(gca, 'FontSize', 8);
end
if strcmp(aggregation_mode, 'minprice')
    sgtitle('Time-averaged m_N(self bin, comp bin) per seller — MIN-of-others', 'FontSize', 14);
else
    sgtitle('Time-averaged m_N(self bin, comp bin) per seller — AVG-of-others', 'FontSize', 14);
end

fig_dir = fullfile(paths.matlab_root, 'output', 'figures', 'application_fixedcost');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
saveas(fig, fullfile(fig_dir, ['mN_heatmap_15sellers' fig_suffix '.png']));
saveas(fig, fullfile(fig_dir, ['mN_heatmap_15sellers' fig_suffix '.pdf']));

%% Save raw mN_all
save(fullfile(paths.matlab_root, 'output', 'application_fixedcost', ['mN_diagnostic' fig_suffix '.mat']), ...
    'mN_all', 'n_sellers', 'n_profiles', 'aggregation_mode');

fprintf('\nSaved heatmap to %s\n', fullfile(fig_dir, ['mN_heatmap_15sellers' fig_suffix '.png']));
fprintf('=== Done. ===\n');
