function [mc_draws] = marginal_cost_draws(cfg, type_space, draws, version)
% DF.SIM.MARGINAL_COST_DRAWS  Unified marginal cost draw function.
%
%   mc_draws = df.sim.marginal_cost_draws(cfg, type_space, draws)
%   mc_draws = df.sim.marginal_cost_draws(cfg, type_space, draws, version)
%
%   Unifies marginal_cost_draws_v4 and marginal_cost_draws_v4_new.
%
%   version:
%     'v4_new' (default) — independent marginals via pdf('normal', ...).
%                          Used by learn_mod / df.sim.learn (Stages II, IV).
%     'v4'               — joint draws via mvnpdf. Used by learn (Stage I).
%
%   cfg must contain: .sigma2, .mu

if nargin < 4 || isempty(version)
    version = 'v4_new';
end

sigma2 = cfg.sigma2;
mu = cfg.mu;

switch version
    case 'v4_new'
        % Independent marginals (from marginal_cost_draws_v4_new)
        if size(sigma2, 1) == 1
            marg_distrib = pdf('normal', type_space{1,1}, mu(1), sigma2(1,1));
            marg_distrib = marg_distrib / sum(marg_distrib);
            mc_draws = randsample(type_space{1,1}, draws, true, marg_distrib);

        elseif size(sigma2, 1) == 2
            combs = [kron(ones(size(type_space{2,1})), type_space{1,1}), ...
                     kron(type_space{2,1}, ones(size(type_space{1,1})))];
            md1 = pdf('normal', type_space{1,1}, mu(1), sigma2(1,1));
            md1 = md1 / sum(md1);
            md2 = pdf('normal', type_space{2,1}, mu(2), sigma2(2,2));
            md2 = md2 / sum(md2);
            joint_distrib = kron(md2, md1);
            mc_inds = randsample(1:size(combs,1), draws, true, joint_distrib);
            mc_draws = combs(mc_inds, :);

        elseif isequal(sigma2, diag(diag(sigma2)))
            mc_draws = zeros(draws, size(sigma2, 1));
            for ind = 1:size(sigma2, 1)
                md_i = pdf('normal', type_space{ind,1}, mu(ind), sigma2(ind,ind));
                md_i = md_i / sum(md_i);
                mc_draws(:, ind) = randsample(type_space{ind,1}, draws, true, md_i);
            end
        else
            error('Arbitrary dependence not implemented');
        end

    case 'v4'
        % Joint draws via mvnpdf (from marginal_cost_draws_v4)
        if size(sigma2, 1) == 2
            combs = [kron(ones(size(type_space{2,1})), type_space{1,1}), ...
                     kron(type_space{2,1}, ones(size(type_space{1,1})))];
            probs = mvnpdf(combs, mu', sigma2');
            inds = (1:size(combs,1))';
            mc_inds = randsample(inds, draws, true, probs);
            mc_draws = combs(mc_inds, :);

        elseif isequal(sigma2, diag(diag(sigma2)))
            for ind = 1:size(sigma2, 1)
                probs = mvnpdf(type_space{ind,1}, mu(ind), sigma2(ind,ind));
                mc_draws(:, ind) = randsample(type_space{ind,1}, draws, true, probs);
            end
        else
            mult = length(mu) + 1;
            discrand = mu' + step_size' .* round(mvnrnd(zeros(size(mu)), sigma2, 1000+mult*draws) ./ step_size');
            bad_inds = sum(abs(discrand - mu') > 3*sqrt(diag(sigma2))', 2);
            discrand(bad_inds > 0, :) = [];
            mc_draws = discrand(1:draws, :);
        end

    otherwise
        error('Unknown version: %s. Use ''v4_new'' or ''v4''.', version);
end

end
