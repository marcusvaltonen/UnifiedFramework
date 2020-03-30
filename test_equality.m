% This experiment shows that the proposed regularizer yields the
% same result as the NN (nuclear norm) and Rmu (from Larsson and Olsson)
% regularizers, when appropriate values for a and b are set.

addpath('helpers')
addpath('solvers_admm_missing_data')

r0 = 5;
sigma = 0;
F = 40;
N = 20;
U = randn(F, r0);
V = randn(N, r0);

Xgt = U * V';
M = Xgt + sigma * randn(size(Xgt));

pattern = 'tracking';
p_missing = 0.5;
W = generate_missing_data_pattern(pattern,p_missing,F,N);
nbr_iter = 10;
X_init = randn(F, N);
quiet = false;

rho = 1.1;
mu = max(F,N);

%% NN relaxation
disp('NN')

w = 2 * sqrt(mu) * ones(min(F, N), 1);
Xsola = WNN_ADMM_MissingData(w, W, M, nbr_iter, rho, X_init, quiet);

a = w / 2;
b = zeros(min(F, N), 1);
Xsol2a = Rh_ADMM_MissingData(a, b, W, M, nbr_iter, rho, X_init, quiet);

%% Rmu relaxation - Larsson and Olsson (2016), IJCV
disp('Rmu')

Xsolc = Rmu_ADMM_MissingData(mu, W, M, nbr_iter, rho, X_init, quiet);

a = zeros(min(F, N), 1);
b = mu * ones(min(F, N), 1);

Xsol2c = Rh_ADMM_MissingData(a, b, W, M, nbr_iter, rho, X_init, quiet);