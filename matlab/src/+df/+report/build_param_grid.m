function [distpars, distribution_parameters] = build_param_grid(mu, sigma2, gridparamM, gridparamV, plot_param)
% DF.REPORT.BUILD_PARAM_GRID  Construct candidate distributional parameter grid.
%
%   [distpars, distribution_parameters] = df.report.build_param_grid(mu, sigma2, gridparamM, gridparamV)
%   [distpars, distribution_parameters] = df.report.build_param_grid(mu, sigma2, gridparamM, gridparamV, plot_param)
%
%   Builds a grid of candidate (mu, sigma) parameters for identification.
%   Unifies the duplicated grid construction from II_MAIN, IV_MAIN, and
%   Identification_Pricing_Game_ApplicationL.
%
%   Inputs:
%     mu          — (NPlayers x 1) mean vector
%     sigma2      — (NPlayers x NPlayers) covariance matrix
%     gridparamM  — (NGridM x 1) mean multipliers (or absolute values)
%     gridparamV  — (NGridV x 1) variance multipliers (or absolute values)
%     plot_param  — 'Both' (default), 'Mean', or 'Variance'
%
%   Outputs:
%     distpars               — (NGrid x D) parameter values; D=1 for 1D, D=2 for 2D
%     distribution_parameters — (3 x NGrid) cell: {dist_name; mu_row; sigma_row}
%
%   For 'Both' mode with simulation (mu is vector):
%     gridparamM entries are multipliers of mu(1)
%     gridparamV entries are multipliers of sigma2(1,1)
%
%   For application mode (mu is scalar):
%     gridparamM entries are absolute mean values
%     gridparamV entries are absolute variance values (multiplied by eye(NPlayer))

if nargin < 5 || isempty(plot_param)
    plot_param = 'Both';
end

NGridM = numel(gridparamM);
NGridV = numel(gridparamV);
NGrid = NGridM * NGridV;

% Detect application mode: scalar mu means absolute values, not multipliers
is_application = isscalar(mu);

if strcmp(plot_param, 'Both')
    distpars = zeros(NGrid, 2);
    distribution_parameters = cell(3, NGrid);

    for ind1 = 1:NGridM
        for ind2 = 1:NGridV
            idx = (ind1 - 1) * NGridM + ind2;
            distribution_parameters{1, idx} = 'Normal';

            if is_application
                % Application mode: gridparamM/V are absolute values
                NPlayer = size(sigma2, 1);
                if NPlayer == 1
                    NPlayer = 2;  % default for application
                end
                distribution_parameters{2, idx} = gridparamM(ind1);
                distribution_parameters{3, idx} = gridparamV(ind2) * eye(NPlayer);
                distpars(idx, :) = [gridparamM(ind1), gridparamV(ind2)];
            else
                % Simulation mode: gridparamM/V are multipliers
                distribution_parameters{2, idx} = gridparamM(ind1) * mu';
                distribution_parameters{3, idx} = gridparamV(ind2) * sigma2';
                distpars(idx, :) = [gridparamM(ind1) * mu(1), gridparamV(ind2) * sigma2(1,1)];
            end
        end
    end

elseif strcmp(plot_param, 'Mean')
    Stepsize = 0.05;
    distpars = zeros(NGrid, 1);
    distribution_parameters = cell(3, NGrid);
    for ind = 1:NGrid
        distribution_parameters{1, ind} = 'Normal';
        distribution_parameters{2, ind} = Stepsize * mu' + Stepsize * (ind - 1) * mu';
        distribution_parameters{3, ind} = sigma2';
        distpars(ind, 1) = Stepsize * mu(1) + Stepsize * (ind - 1) * mu(1);
    end

elseif strcmp(plot_param, 'Variance')
    distpars = zeros(NGrid, 1);
    distribution_parameters = cell(3, NGrid);
    for ind = 1:NGrid
        distribution_parameters{1, ind} = 'Normal';
        distribution_parameters{2, ind} = mu';
        distribution_parameters{3, ind} = gridparamV(ind) * sigma2';
        distpars(ind, 1) = gridparamV(ind) * sigma2(1,1);
    end

else
    error('Unknown plot_param: %s. Use ''Both'', ''Mean'', or ''Variance''.', plot_param);
end

end
