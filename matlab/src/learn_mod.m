function [distY_time, distY_time_obs, final_regret, Pl1_EmpRegr, Pl2_EmpRegr] = learn_mod(cfg, N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2)
% learn_mod  Regret-matching learning simulation.
%
%   [distY_time, distY_time_obs, final_regret, Pl1_EmpRegr, Pl2_EmpRegr] = ...
%       learn_mod(cfg, N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2)
%
%   cfg must contain: .A, .AA, .NPlayers, .alpha, .type_space, .learning_style

% Unpack cfg fields used in inner loops
A = cfg.A;
AA = cfg.AA;
NPlayers = cfg.NPlayers;
alpha = cfg.alpha;
type_space = cfg.type_space;
learning_style = cfg.learning_style;

% Initialize objects
s = size(type_space{1,1}, 1);  % number of possible marginal cost realizations
U_n = zeros(s, NPlayers);  % average utility for each type and player
Avg_cf_util = zeros(size(AA,1), s, NPlayers);  % average counterfactual utility
type_counts = zeros(s, NPlayers);  % count of each type occurrence

% Initialize distribution trackers
distY_time = zeros(size(A,1), numdst_t);
distY_time_obs = zeros(size(A,1), numdst_t_obs);
action_counts = zeros(size(A,1), 1);
final_regret = zeros(s,NPlayers);
Pl1_EmpRegr = zeros(s,1);
Pl2_EmpRegr = zeros(s,1);

for t = 1:(N+M)
    % Draw marginal costs for this period
    mc_draw = marginal_cost_draws_v4_new(cfg, type_space, 1);

    % Determine the type index for each player
    type_indices = zeros(1, NPlayers);
    for j = 1:NPlayers
        type_indices(j) = find(type_space{j,1} == mc_draw(j));
    end

    if t <= N
        % Random play for the first N actions
        action_indices = randi(size(A, 1), 1, 1);
        actions = A(action_indices, :);
    else
        % Choose actions using regret matching
        actions = zeros(1, NPlayers);
        for j = 1:NPlayers
            if strcmp(learning_style, 'rm')
                [actions(j), ~] = regret_matching_mod(AA, U_n(:,j), Avg_cf_util(:,:,j), type_indices(j));
            elseif strcmp(learning_style, 'fp')
                error('Fictitious play not implemented in this version');
            else
                error('You have failed to specify an appropriate learning style. Please choose either "fp" or "rm"');
            end
        end
    end

    % Update U_n and Avg_cf_util
    for j = 1:NPlayers
        type_j = type_indices(j);
        type_counts(type_j, j) = type_counts(type_j, j) + 1;

        % Calculate utility for the chosen action
        utility = choice_prob(actions(j), actions, j, alpha) * (actions(j) - mc_draw(j));

        for tt = 1:s

            if tt == type_j
            % Update U_n FOR ALL TYPES
            U_n(tt, j) = ((t-1) * U_n(tt, j) + utility) / t;

            % Update Avg_cf_util FOR ALL TYPES
            for a = 1:size(AA,1)
                cf_utility = choice_prob(AA(a), actions, j, alpha) * (AA(a) - mc_draw(j));
                Avg_cf_util(a, tt, j) = ((t-1) * Avg_cf_util(a, tt, j) + cf_utility) / t;
            end

            else
                    U_n(tt, j) = ((t-1) * U_n(tt, j)) / t;
                    for a = 1:size(AA,1)
                    Avg_cf_util(a, tt, j) = ((t-1) * Avg_cf_util(a, tt, j)) / t;
                    end
            end
    end
    end

    % Update action counts for distribution calculation
    action_index = find(all(A == actions, 2));
    action_counts(action_index) = action_counts(action_index) + 1;

    % Calculate and store distributions at specified intervals
    if ismember(t, round(M * (1:numdst_t)/numdst_t))
        distY_time(:, find(round(M * (1:numdst_t)/numdst_t) == t)) = action_counts / t;
    end

    if t > N+M-M_obs && ismember(t-(N+M-M_obs), round(M_obs * (1:numdst_t_obs)/numdst_t_obs))
        distY_time_obs(:, find(round(M_obs * (1:numdst_t_obs)/numdst_t_obs) == t-(N+M-M_obs))) = ...
            action_counts / t;
    end

    if t == N+M
        for j = 1:NPlayers
            for type_j = 1:s
                [~, final_regret(type_j,j)] = regret_matching_mod(AA, U_n(:,j), Avg_cf_util(:,:,j),type_j);
            end
        end
    elseif t==N+Nobs_Pl1
            for type_j = 1:s
                [~, Pl1_EmpRegr(type_j)] = regret_matching_mod(AA, U_n(:,1), Avg_cf_util(:,:,1),type_j);
            end
    elseif t==N+Nobs_Pl2
            for type_j = 1:s
                [~, Pl2_EmpRegr(type_j)] = regret_matching_mod(AA, U_n(:,2), Avg_cf_util(:,:,2),type_j);
            end
    end

end

end
