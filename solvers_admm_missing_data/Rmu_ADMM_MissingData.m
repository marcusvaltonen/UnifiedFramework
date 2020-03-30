function X = Rmu_ADMM_MissingData(mu, W, M, iter, rho, X, quiet)

Lambda = zeros(size(W));

for i = 1:iter
    Y = lsqLinMap(X, Lambda, W, M, rho);
    X = proxR(mu, Y, Lambda, rho);
    Lambda = Lambda + X - Y;

    if ~quiet
        [reg, datafit] = ObjectiveValue(X, mu, W, M);
        objval = reg + datafit^2;
        fprintf(1,'iter: %3d\tReg %f\tData %f\tObjective value %f\tVariable difference %f\n', ...
            i, reg, datafit, objval, norm(X - Y,'fro'));
    end
    if norm(X - Y, 'fro') < 1e-8
       return 
    end
end

function [reg, datafit] = ObjectiveValue(X,mu,W,M)
sX = svd(X, 'econ');
reg = sum(mu - max(sqrt(mu) - sX, 0).^2);
datafit = norm(W .* (X-M), 'fro');

function Y = lsqLinMap(X, Lambda, W, M, rho)
Y = (rho * (X + Lambda) + W.* M) ./ (rho + W);

function X = proxR(mu, Y, Lambda, rho)
M = Y-Lambda;
[U,S,V] = svd(M,'econ');
m = diag(S);

ind1 = m <= sqrt(mu) / rho;
ind2 = sqrt(mu) / rho < m & m <= sqrt(mu);

m(ind1) = 0;
m(ind2) = (rho * m(ind2) - sqrt(mu)) / (rho - 1);

X = U * diag(m) * V';