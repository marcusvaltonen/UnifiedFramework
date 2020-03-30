function [ X, x, syvec ] = proxRh(avec,bvec,X0,P,rho)
% Solves the problem
%   min_X R_h(X) + |X-X0|^2 + rho * |X-P|^2

Y = (X0+rho*P)/(1+rho);
[U,S,V] = svd(Y,'econ');
syvec = diag(S);

% Seq. of unconstrained minimizers
si = zeros(size(avec));
for j = 1:length(avec)
    a = avec(j);
    b = bvec(j);
    sy = syvec(j);
    
    asb = a+sqrt(b);
    
    if a/(rho+1) + sqrt(b) < sy
        si(j) = a * rho / (rho+1) + sy;
    elseif asb/(1+rho)<= sy &&  sy <= a/(rho+1) + sqrt(b)
      	si(j) = asb;
    elseif sy < asb/(1+rho)
        si(j) = (1+rho)*sy;
    else
        warning('something went wrong.')
    end
    
end

% Find maximizing pseudo-singular vector
x =  maximizing_sequence_prox(avec,bvec,syvec,si,rho);

% Construct solution
Z = U*diag(x)*V';
X = P + (X0-Z)/rho;

function sigma = maximizing_sequence_prox(a,b,sy,si,rho)

sigma = si;

options = optimset('Display','none');  % FMINBND IS FASTER THAN FMINUNC

cnt = 0;

while ~issorted(sigma,'descend')
    
    % Find regions of same singular value
    mins = find(islocalmin(sigma, 'FlatSelection', 'first'));
    maxs = find(islocalmax(sigma, 'FlatSelection', 'last'));
    
    % TODO: Check this logic, it might not cover all cases
    if ~isempty(maxs)
        if isempty(mins) || mins(1) > maxs(1)
            % In this case the sequence starts out increasing
            mins = [1 mins];
        end
    end
    if (~isempty(mins) && isempty(maxs)) || mins(end)>maxs(end)
        maxs = [maxs length(sigma)];
    end
    
    if length(mins) ~=length(maxs)
        warning('Something went wrong')
    end
    
    intervals = [mins; maxs]';
    
    % Maximize each interval separately
    for j = 1:size(intervals,1)
        n = diff(intervals(j,:)) + 1;   % Nbr of points in interval
        idx = intervals(j,1):intervals(j,2); % Indices of points in the interval
        ss = @(x) x*ones(1,n);
        objfunc = @(x) singZ_prox(a(idx), b(idx), sy(idx), ss(x), rho);
        % TODO: Maybe this can be solved analytically - or at least make
        % something faster by splitting in subintervals
        x_out = fminbnd(objfunc,si(intervals(j,1)),si(intervals(j,2)),options);
        sigma(idx) = x_out;
    end
    cnt = cnt + 1;
end

function val = singZ_prox(a, b, sy, s, rho)
N = length(a);
val = 0;
for j=1:N
    val = val + max(max(s(j)-a(j),0)^2-b(j),0) + 1/rho*s(j)^2 - 2*(rho+1)/rho*s(j)*sy(j) + (rho+1)/rho*sy(j)^2;
end