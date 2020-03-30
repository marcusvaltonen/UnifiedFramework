function X = Rh_ADMM_MissingData(a,b,W,M,iter,rho,X,quiet)

Lambda = zeros(size(W));

for i = 1:iter
    Y = lsqLinMap(X,Lambda,W,M,rho);
    [X,sZ] = proxRh(a,b,Y-Lambda,Y-Lambda,rho-1);
    Lambda = Lambda+X-Y;

    if ~quiet   
        [reg, datafit] = ObjectiveValue(X,a,b,W,M,sZ);
        objval = reg + datafit^2;
        fprintf(1,'iter: %3d\tReg %f\tData %f\tObjective value %f\tVariable difference %f\n', ...
            i, reg, datafit, objval, norm(X - Y,'fro'));
    end
    if norm(X - Y,'fro') < 1e-8
        return
    end
end

function [reg,datafit] = ObjectiveValue(X,a,b,W,M,sZ)
sX = svd(X,'econ');
reg = EnergyReg(sX,a,b,sZ);
datafit = norm(W.*(X-M),'fro');

function regval = EnergyReg(sX,a,b,sZ)
regval = 0;

for j=1:length(a)
    regval = regval - max(max(sZ(j)-a(j), 0).^2 - b(j), 0) - (sX(j)-sZ(j))^2 + sZ(j)^2;
end

function Y = lsqLinMap(X, Lambda, W, M, rho)
Y = (rho * (X + Lambda) + W.* M) ./ (rho + W);
