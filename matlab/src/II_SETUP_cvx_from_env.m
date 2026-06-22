%% II_SETUP_cvx_from_env
%
% Run cvx_setup from the CVX_DIR environment variable.

clear; clc;

cvx_dir = getenv('CVX_DIR');
assert(~isempty(cvx_dir), 'Set CVX_DIR before running II_SETUP_cvx_from_env.');
assert(exist(cvx_dir, 'dir') == 7, 'CVX_DIR does not exist: %s', cvx_dir);

old_dir = pwd;
cleanup_obj = onCleanup(@() cd(old_dir));

fprintf('Running cvx_setup from %s\n', cvx_dir);
cd(cvx_dir);
cvx_setup;

fprintf('cvx_setup finished.\n');
