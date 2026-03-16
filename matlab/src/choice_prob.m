function [cp] = choice_prob(a,b,j,alpha)

%==========================================================================
%FUNCTION: choice_prob
%AUTHOR: Jonathan E. Becker
%DESCRIPTION: Choice probabilities
%--------------------------------------------------------------------------
%INPUTS:
%1. a | Vector of actions (Use column vector for past choice probabilities, 
%       row vector for counterfactual choice probabilities)
%2. b | Matrix of all past actions
%3. j | Scalar index of the current player
%4. alpha | Price parameter (should be negative)
%OUTPUTS: 
%1. cp | Matrix of choice probabilities for actions in a given past actions
%        in b
%==========================================================================

% Check Dimension of 'a'
if length(a)==1
    a = repmat(a,[size(b,1),1]);
end

% Calculate Choice Probabilities
denom = 1 + sum(exp(alpha*b),2) + exp(alpha*a) - exp(alpha*b(:,j));
cp = exp(alpha*a)./denom;
