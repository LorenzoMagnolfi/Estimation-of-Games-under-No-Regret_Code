function [distY_time, distY_time_obs, actions, regret, type_inds] = learn(N,M,M_obs,numdst_t,numdst_t_obs)
%% Initialize Distributions

global A AA NPlayers alpha type_space learning_style 

distY_time = zeros(4,numdst_t);
distY_time_obs = zeros(4,numdst_t_obs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define this as a function??

% A combs
Acombs = [kron(ones(size(AA)),AA), kron(AA,ones(size(AA)))];
inds = (1:size(Acombs,1))';
% probs = simplex_draws(1,101,4,false);
r = unifrnd(0,1,1,size(A,1));
probs = r/sum(r);
sample_inds = randsample(inds,N,true,probs);

% Preallocate Space 
actions = zeros(N+M,NPlayers);                                                   % Zeros for actions
regret = zeros(N+M,NPlayers);                                                   % Zeros for actions

%% Initial Condition
actions(1:N,:) = Acombs(sample_inds,:);

%Draw MCs
[mc] = marginal_cost_draws_v4(type_space,N+M);

%% Learning Phase
for ind=1:M
    % Regret
    for j=1:NPlayers
        if strcmp(learning_style,'rm')
            [actions(N+ind,j), regret(N+ind,j)] = regret_matching_new(AA,actions(1:N+ind-1,:),mc(1:N+ind-1,:),mc(N+ind,:),j,alpha,ind);
            % actions(N+ind,j) = regret_matching(AA,actions(1:N+ind-1,:),mc(1:N+ind-1,:),mc(N+ind,:),j,alpha);
        elseif strcmp(learning_style,'fp')
            actions(N+ind,j) = fictitious_play(AA,actions(1:N+ind-1,:),mc(N+ind,:),j,alpha);
        else
            error('You have failed to specify an appropriate learning style. Please choose either "fp" or "rm"');
        end
    end

    % Display Empirical Joint Distribution (50 Updates)
    if mod(ind,round(M/50)) == 0
        % Display Progress
        disp(['Round ' num2str(ind) ' of ' num2str(M)])
        
        % Data Visualization
        if length(AA) > 20 && NPlayers == 2
            % Only display distribution where both players participate
            disp_inds = sum(actions,2)<1e20;
            disp_inds = disp_inds(1:N+ind);
            
            % Joint Distribution
            ksdensity(actions(disp_inds,:));
        else
            % Display OFF: 
            % ksdensity is going to look ugly if the action space is not sufficiently fine-grained
        end
        drawnow;
        
    end
    
end


%% Save joint distribution of actions at various points
% Grid of joint actions

% Construct Distributions
distY_time = zeros(size(A,1),numdst_t);
ind1 = 1;
for obs = round(M * (1:numdst_t)/numdst_t )
    for ind2 = 1:size(A,1)
        distY_time(ind2,ind1) = sum(prod(actions(1:(N+obs),:) == A(ind2,:),2))/(N+obs); 
    end
    ind1 = ind1 + 1;
end

% Construct Distributions that are Observed by the Econometrician
% These are the computed using the last M_obs observations
ind1_obs = 1;
for obs2 = round(M_obs * (1:numdst_t_obs)/numdst_t_obs )
    for ind2_obs = 1:size(A,1)
            distY_time_obs(ind2_obs,ind1_obs) = sum(prod(actions(N+M-M_obs:N+M-M_obs+obs2,:) == A(ind2_obs,:),2))/(obs2); 
    end
    ind1_obs = ind1_obs+1;
end


type_inds(:,:,1) = mc(:,1) == type_space{1,1}';
type_inds(:,:,2) = mc(:,2) == type_space{2,1}';


end
