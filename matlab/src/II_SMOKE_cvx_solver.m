%% II_SMOKE_cvx_solver
%
% Tiny server smoke test for CVX and SeDuMi before launching production
% simulations. This does not run any paper simulation.

clear; clc;

II_CHECK_corrected_bundle_prereqs;

cvx_clear;
cvx_solver sedumi;
cvx_begin quiet
    variable x
    minimize((x - 1)^2)
cvx_end

fprintf('CVX status: %s\n', cvx_status);
fprintf('x: %.12g\n', x);

assert(any(strcmp(cvx_status, {'Solved', 'Inaccurate/Solved'})), ...
    'CVX smoke test did not solve successfully.');
assert(abs(x - 1) < 1e-5, 'CVX smoke-test solution is not close to 1.');

fprintf('CVX smoke test passed.\n');
