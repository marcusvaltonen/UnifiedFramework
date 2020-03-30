function W = generate_missing_data_pattern(pattern,p_missing,m,n)

if strcmp('uniform',pattern)
    W = rand(m,n) > p_missing;
elseif strcmp('tracking',pattern)
    W = tracking_pattern(p_missing,m,n);
elseif strcmp('tracking2',pattern)
    W1 = tracking_pattern(p_missing,m,n);
    W2 =  rot90(tracking_pattern(p_missing,m,n), 2);
    W = W1 & W2;
elseif strcmp('block-diag',pattern)
    W = true(m,n);
    ind_miss = rand(1,n) < p_missing;
    for j = find(ind_miss)
        fail_point = randi([floor(0.1*m) m]);
        W(fail_point:end,j) = 0;
    end
    [~,I] = sort(sum(W,1));  
    W = W(:,I);
    
    W2 = true(m,n);
    ind_miss = rand(1,n) < p_missing;
    for j = find(ind_miss)
        fail_point = randi([floor(0.1*m) m]);
        W2(fail_point:end,j) = 0;
    end
    [~,I] = sort(sum(W2,1));
    W2 = W2(:,I);
    
    W = W & W2';
else
    error('Incorrect pattern')
end
end

function W = tracking_pattern(p_missing,m,n)
    % Tracking failures
    W = true(m,n);
    ind_miss = rand(1,n) < p_missing;
    for j = find(ind_miss)
        fail_point = randi([floor(0.1*m) m]);
        W(fail_point:end,j) = 0;
    end
    [~,I] = sort(sum(W,1));
    W = W(:,I);
end
