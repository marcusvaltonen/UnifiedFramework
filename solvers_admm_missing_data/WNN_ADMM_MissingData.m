function X = WNN_ADMM_MissingData(w, W, M, iter, rho, X, quiet)
Lambda = zeros(size(W));

for i = 1:iter
    Y = lsqLinMap(X, Lambda, W, M, rho);
    X = proxR(w, Y, Lambda, rho);    
    Lambda = Lambda + X - Y;

    if ~quiet
        [reg, datafit] = ObjectiveValue(X, w, W, M);
        objval = reg + datafit^2;
        fprintf(1,'iter: %3d\tReg %f\tData %f\tObjective value %f\tVariable difference %f\n', ...
            i, reg, datafit, objval, norm(X - Y,'fro'));
    end
    if norm(X - Y,'fro') < 1e-8
       return 
    end
end

function [reg, datafit] =  ObjectiveValue(X, w, W, M)
reg = sum(w .* svd(X, 'econ'));
datafit = norm(W .* (X - M), 'fro');

function X = proxR(mu, Y, Lambda, rho)
[U, S, V] = svd(Y - Lambda, 'econ');
sM = diag(S);

sX = sM - mu / 2 / rho;
sX(sX < 0) = 0;

S(1:length(sM), 1:length(sM)) = diag(sX);
X = U * S * V';

function Y = lsqLinMap(X, Lambda, W, M, rho)
Y = (rho * (X + Lambda) + W .*  M) ./ (rho + W);