function r = compute_consistency_slack(M_C, alpha_C, N, kind)
% COMPUTE_CONSISTENCY_SLACK  Sampling-error slack on the consistency constraint.
%
%   r = df.solvers.compute_consistency_slack(M_C, alpha_C, N)
%   r = df.solvers.compute_consistency_slack(M_C, alpha_C, N, kind)
%
%   Per Niccolo's "convergence rates" note: when finite-sample inference
%   takes the consistency block seriously, the empirical (t,theta)-marginal
%   m_N(t,theta) fluctuates around the candidate prior p_lambda(t,theta)
%   at a sqrt(N)-rate.  The size of the slack depends on which functional
%   we bound.
%
%   Inputs
%     M_C      |T| * |Theta|, the number of (type-profile, state) cells.
%     alpha_C  consistency-block confidence budget (alpha_C + alpha_R = alpha).
%     N        sample size.
%     kind     'box'  (default) — coordinatewise Hoeffding bound:
%                                 max_{t,theta} |m_N - p| <= r,
%                                 r = sqrt(log(2 M_C / alpha_C) / (2 N))
%              'L1'   — joint Bretagnolle-Huber-Carol L1 bound:
%                                 sum_{t,theta} |m_N - p| <= r,
%                                 r = sqrt(2 log((2^M_C - 2) / alpha_C) / N)
%
%   Output
%     r        nonneg scalar slack.

if nargin < 4 || isempty(kind)
    kind = 'box';
end

switch kind
    case 'box'
        r = sqrt(log(2 * M_C / alpha_C) / (2 * N));
    case 'L1'
        % SIM-6: log-domain to avoid 2^M_C overflowing to Inf for large M_C.
        % log(2^M_C - 2) = M_C*log(2) + log1p(-2^(1-M_C)); the correction term
        % underflows harmlessly to 0, leaving M_C*log(2) for large M_C.
        log_num = M_C * log(2) + log1p(-2^(1 - M_C));
        r = sqrt(2 * (log_num - log(alpha_C)) / N);
    otherwise
        error('compute_consistency_slack:badKind', ...
            'kind must be ''box'' or ''L1'', got ''%s''.', kind);
end

end
