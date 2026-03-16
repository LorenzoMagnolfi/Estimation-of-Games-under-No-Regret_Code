function [outputs] = ComputeBCCE_eps_ApplicationL(type_space,action_space,action_distribution,payoff_parameters,distribution_parameters,T,confid,Pi,switch_eps,marg_mean,cfg)

%==========================================================================
%--------------------------------------------------------------------------
%INPUTS:
%1. type_space | Column Cell Matrix 
%   - Each entry gives the type space for an individual
%2. action_space | Column Cell Matrix
%   - Each entry gives the action space for an individual
%4. covariates | Matrix of scalars
%   - Each row gives a set of covariates which potentially affect payoffs
%5. action_distribution | Matrix of scalars
%   - Each row vectors gives joint distribution of actions 
%6. payoff_parameters | 
%   - Payoff parameters to check 
%7. distribution_parameters | Cell Vector: {1: String; 2-...: Parameters}
%   - Distributional parameters to check
%OUTPUTS: 
%1. outputs | Matrix of scalars
%   - Objective function values for distribution/payoff parameters
%--------------------------------------------------------------------------
%NOTES: This code builds on the Lorenzo Magnolfi and Camilla Roncoroni for
%the two-player entry game.
%==========================================================================

%% Problem Dimensions
% Number of marginal cost distribution parameters
% This is how large the grid on the variance of mc parameters (lambda) is
NGrid_lambda = size(distribution_parameters,2);

% If empty, we'll just use a uniform prior, but we need ND to be positive
if NGrid_lambda == 0
    NGrid_lambda = 1;
end

% Number of observed distributions q^N, for different values of N
Nq_N = size(action_distribution,2);

% Number of different demand parameters alpha - we keep it fixed at 1
% MAYBE get rid of this?
NAlpha = size(payoff_parameters,1);

% Number of Agents
NAg = size(type_space,1);

% Number of Actions (A_i) and Types (T_i) for each agent i
for ind=1:NAg
    NA_i(ind,1) = size(action_space{ind,1},1);
    NT_i(ind,1) = size(type_space{ind,1},1);
end

Nactions = NA_i(1); 

% Number of possible action profiles
NA = prod(NA_i);

% Number of type profiles
s2 = prod(NT_i);

% since NT_i are the same
s = NT_i(1);

% Dimension of the vectorized BCE Probability Measure
%dv = s * NA;
dv = s * Nactions;

% Equality Constraint: One constraint for each unique type and action, plus one for the condition that the measure integrates to one
%deq = 1 + NA + s;
deq = 1 + Nactions + s;

% Inequality Constraint: For each type, one for each combination of two actions
% dineq = NT_i'*(NA_i.*(NA_i-1));
% dineq = NT_i'*(NA_i)/2;
dineq = s*Nactions;

% Full Constraint Matrix: Dim(Equalities) + Dim(Inequalities)
dM = deq + dineq;

% dimension of the w variable of the convex program
% dw = NA - 1 + deq + dineq;
dw = Nactions - 1 + deq + dineq;

%%  Sorted Type Profiles
% All Possible type profiles, ordered as in the computational Appendix

% define a dummy useful for reshaping 
AEnum = 1;

% initialize
T_sorted = type_space{1,1};

for ind=2:NAg
    AEnum = [size(T_sorted,1), AEnum];
    T_sorted = [kron(type_space{ind,1}, ones(size(T_sorted,1),1) ) ...
    kron( ones(size(type_space{ind,1},1),1), T_sorted)]; 
end

%% Sorted Action/Type Profiles
% Grid is each combination of actions/types for all players: (a1,...,aN,t1,...,tN)
% Start with type space grid

AT_sorted = T_sorted;

% Go through actions, one individual at a time
for ind = 1:NAg
    % Number of combinations (used later)
    AEnum = [size(AT_sorted,1), AEnum];
    % Add actions, starting with all combinations of yN
    AT_sorted = [kron(action_space{ind,1}, ones(size(AT_sorted,1),1) ) ...
         kron(ones(size(action_space{ind,1},1),1), AT_sorted)];
end

%% Compute Common Prior Over Payoff States
% Notice the peculiar definition of truncated normal...

% Initialize the matrix of distributions psi for all values of lambda
Psi = zeros(s2,NGrid_lambda); 

if size(distribution_parameters,1) == 0
    Psi = ones(s2,1);
elseif size(distribution_parameters,1) == 3
    for nd = 1:NGrid_lambda
        if strcmp(distribution_parameters{1,nd},'Normal')
            Psi(:,nd) = mvnpdf(T_sorted,distribution_parameters{2,nd},distribution_parameters{3,nd});
        else
            Psi(:,nd) = pdf(distribution_parameters{1,nd},T_sorted,distribution_parameters{2,nd},distribution_parameters{3,nd});
        end 
    end
end

% Normalize
% Psi = Psi./sum(Psi,1);

%% Alternative Psi done with the product of marginals

for nd = 1:NGrid_lambda
    mean = distribution_parameters{2,nd};
    mean = mean(1);
    sigma = distribution_parameters{3,nd};
    sigma = sigma(1,1);
    marg_distrib(:,nd) = pdf(distribution_parameters{1,nd},type_space{1,1},mean,sigma);
    marg_distrib(:,nd) = marg_distrib(:,nd)./sum(marg_distrib(:,nd),1);
    
    Psi_new(:,nd) = kron(marg_distrib(:,nd),marg_distrib(:,nd));
end

Psi = marg_distrib;

%% Manipulate Data
marg_act_distrib_I = kron(eye(NA_i(1)),ones(1,NA_i(1)))*action_distribution;
marg_act_distrib_II = kron(ones(1,NA_i(1)),eye(NA_i(1)))*action_distribution;

%% Boundary Conditions

% Lower and upper bounds for our variables. In addition to the constraints 
% we have to impose (on lambda_ineq>=0), we bound b_tilde for tractability 
%Comp_lb = [-1*ones(NA-1,1);  -Inf*ones(deq,1); zeros(dineq,1)];
Comp_lb = [-1*ones(Nactions-1,1);  -Inf*ones(deq,1); zeros(dineq,1)];
lb0 = Comp_lb;  %lower bound
%Comp_ub = [1*ones(NA-1,1);  Inf*ones(dM,1)];
Comp_ub = [1*ones(Nactions-1,1);  Inf*ones(dM,1)];

ub0 = Comp_ub;  %upper bound
clear Comp_lb Comp_ub

% This is the bound we add, to avoid "Inf"
lb = max(lb0,-10000);
ub = min(ub0,10000);
clear lb0 ub0

%% Equality Constraints 
% To construct matrices of linear constraints, first assemble matrix A

%M1eq = kron(ones(1,s),eye(NA));
M1eq = kron(ones(1,s),eye(Nactions));
% M2eq = kron(eye(s),ones(1,NA));
M2eq = kron(eye(s),ones(1,Nactions));

%Meq = [eye(NA), M1eq; ...
%    zeros(s,NA), M2eq; ...
%    zeros(1,NA), ones(1,dv)];
Meq = [eye(Nactions), M1eq; ...
    zeros(s,Nactions), M2eq; ...
    zeros(1,Nactions), ones(1,dv)];

%% Other Fixed Objects
% vectors of constants on the RHS of equality and inequality constraints
% beq = zeros(NA,1);
beq = zeros(Nactions,1);
b = zeros(dv,1);  

% Nonlinear Constraints: The only nonlinear constraint is norm(Mat_NLC*x)<=1
% Mat_NLC = [eye(NA-1) zeros(NA-1,dM)]; 
Mat_NLC = [eye(Nactions-1) zeros(Nactions-1,dM)]; 

%% Now construct Inequality constraints

% C Objects
C1=eye(s);   

%% NEW WORK on M1ineq for 10 actions!

% s = 3; %number of payoff types of each player
A = NA; % number of action profiles
a = Nactions;
pi_1 = reshape(squeeze(Pi(:,:,1)),NA*s,1); % payoff vectors

%% new work on EPS

util = Pi;

% NOW do expected payoffs:
exp_pi = kron(eye(Nactions),marg_act_distrib_II') * util(:,:,1);

% EPS linked to rate of convergence!
eps = epsilon_switch_distrib(T,confid,switch_eps,marg_mean,s,cfg);

%% Define basic objects

% pi1_res = reshape(pi_1,A,s); % reshape pi so that every row is act profile, every column type
pi1_res = reshape(exp_pi,Nactions,s); % reshape pi so that every row is act profile, every column type

% pi_tilde_1T = kron(pi_1',ones(s,1)');
% pi_tilde_1T = pi_1';
pi_tilde_1T = exp_pi';


%% Betas

%NEW!
Beta_1 = kron(C1,ones(Nactions,1)') .* kron(ones(1,s),pi_tilde_1T);

%% Alphas!
%{
%% Alpha 1
E= eye(a);

alpha_1 = zeros(s,s*A,a);

% First reshape!
pi_1_hat = reshape(pi_1,A,s);


for j = 1:a

% Idea: do all of these as if eps are an afterthought; then put all
% together (i.e., do eps by eps by reshaping vector)

% This reshape is straightforward, as we are taking pieces corresponding to
% different eps - in line with ordering of matrix/vectors
pi_j_1 = reshape(kron(ones(a,1),kron(E(:,j)',eye(a)))*pi_1_hat,A*s,[]);

% Have to take care of the tildas!
% pi_tilde_j_1T = kron(pi_j_1',ones(s,1)');
pi_tilde_j_1T = reshape(repmat(reshape(pi_j_1,A,s),1,1),A*s,1)';

% Now Alphas
alpha_j_1 = kron(C1,ones(A,1)') .* kron(ones(s,1),pi_tilde_j_1T);

alpha_1(:,:,j) = alpha_j_1;

end

% Now reshape appropriately the alpha_1 and alpha_2 matrix; see 
% https://stackoverflow.com/questions/32810010/reshape-matrix-from-3d-to-2d-keeping-specific-order
alpha_1_dev = reshape(permute(alpha_1,[1 3 2]),[],size(alpha_1,2));
%}

reshape(pi1_res,s*Nactions,1)
EE = eye(s);

aaa=1;
while aaa <= Nactions
    for jj = 1:s
        alpha_1_dev(((aaa-1)*s)+jj,:) = pi_tilde_1T(jj,aaa)*kron(EE(jj,:),ones(1,Nactions));
    end
    aaa=aaa+1;
end

%% NOW assmble all in deviation matrix!

% M1ineq = [alpha_1_dev] - [kron(ones(a,1),Beta_1)];
M1ineq = [alpha_1_dev] - [kron(ones(Nactions,1),Beta_1)];

%Mineq = [zeros(dineq,NA), M1ineq];
Mineq = [zeros(dineq,Nactions), M1ineq];

%% Find Optimal Parameters
% Collect values for every parameter and x
g = zeros(NGrid_lambda,Nq_N);

% Check all parameter combinations
for nd = 1:NGrid_lambda

% Fix the prior
bmarg = Psi(:,nd);
        
    
    %% Check all observations (different true parameters + distributions) 
    % OBJ for minimization  - enters as vecObj*vars (u, dual2,3,4,5)
    for nb=1:Nq_N
        
    %% Prepare the Dual Problem
    % Combine Equalities and Inequalities in a single matrix
    M = [Meq; ...
        Mineq];     % this is for Ax<=b
    M_prime = M';
        
    % Dual problem equailities matrix
    B_EQ_comp = [eye(Nactions-1); 
                 zeros(1,Nactions-1)];
    B_EQ = [B_EQ_comp, M_prime(1:Nactions,:)];
    
    % Dual problem inequalities matrix
    B_INEQ = [zeros(dv,(Nactions-1)), M_prime((Nactions+1):end,:)] ; % CHECK PDF!!
    
    if switch_eps==1 || switch_eps==3
    %eps_fin = repmat(marg_distrib(:,nd)',1,a).*repmat(sqrt(marg_distrib(:,nd))',1,a).*repmat(eps,1,a);
    eps_fin = repmat(sqrt(marg_distrib(:,nd))',1,a).*repmat(eps,1,a);
    else
    %eps_fin = repmat(marg_distrib(:,nd)',1,a).*repmat(eps,1,a);    
    eps_fin = repmat(eps,1,a);    
    end

    %eps_fin3 = repmat(marg_distrib(:,nd)',1,a).*repmat(sqrt(marg_distrib(:,nd))',1,a).*repmat(eps3,1,a);
    %eps_fin6 = repmat(marg_distrib(:,nd)',1,a).*repmat(eps6,1,a);
    %eps_fin7 = repmat(marg_distrib(:,nd)',1,a).*repmat(eps7,1,a);
    
    %mean(eps_fin3)
    %mean(eps_fin7)
    
    c = [zeros(1,Nactions-1), marg_act_distrib_I', bmarg', 1, eps_fin]';
   
    % Size of the problem
    n = size(c,1);
    
    %% Convex Optimization Solution (Using CVX Solver)
    % Choice of Solver:
    cvx_solver sedumi     % For SeDuMi
    % cvx_solver mosek      % For MOSEK
    % cvx_solver sdpt3      % For SDPT3
    tic
    % CVX Problem
    cvx_clear
    cvx_begin 
        % Variable
        variable x(n);
        dual variables lambdaEq lambdaIneq lambdaBds lambdaNorm
        
        % Optimization problem
        maximize( - c' * x)

        % Constraints
        subject to 
        lambdaEq: B_EQ * x == beq;
        lambdaIneq: B_INEQ * x >= b;
        lambdaBds: lb <= x <= ub;
        
        % Plus the nonlinear constr
        lambdaNorm: norm(Mat_NLC * x) <= 1;
    cvx_end
    
    % Retain Values
    if strcmp(cvx_status,"Solved")
    g(nd,nb) = cvx_optval;
    else
    g(nd,nb) = 100;    
    end 
    
    % Display Progress
    disp(['Distribution: ' num2str(nd) '/' num2str(NGrid_lambda) ' Observation: ' num2str(nb) '/' num2str(Nq_N)]);
    toc
    end


end
    
%% Outputs
% Output Optimal Value Matrix

outputs{1,1} = g;

% Check other problem parameters (e.g., inequality + equality constraints)
% outputs{2,1} = A;
end