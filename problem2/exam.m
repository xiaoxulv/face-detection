W = [0.3 0.5 0.7 0.6 0.2 -0.8 0.4 0.2;-0.6 -0.5 -0.1 -0.4 0.4 -0.1 -0.9 0.5] ;
total = size(W,2);
wunit = 1/total;
weight = repmat(wunit,total,1);
real = [1 1 1 1 0 0 0 0];

round = 10;
alpha = zeros(round,1);
mineigen = zeros(round,1);
minthresh = 100;
minthresh = repmat(minthresh,round,1);
sign = ones(round,1); 

for r = 1:round
    minerr = 100;
    minapply = zeros(total,1);
    for i = 1:2 % for 10 eigenfaces
        maxV = max(W(i,:));
        minV = min(W(i,:));
        unit1 = (maxV-minV)/30;
        for k = 1:29 % for 100 gaps
            b = 0;
            thresh = minV + k*unit1;
            % predict classification for all
            apply = zeros(total,1);
            err = 0;
            for j = 1:total %for all the training data
                if W(i,j) > thresh
                    apply(j) = 1;
                end
                % error weight sum
                if apply(j) ~= real(j)
                    err = err + weight(j);
                end
            end
            % error rate control
            if err > 0.5
                err = 1 - err;
                b = 1;
                apply~=1;
            end
            % update minimum error
            if err <= minerr
                minerr = err;
                minthresh(r) = thresh;
                mineigen(r) = i;
                minapply = apply;
                if b == 1
                    sign(r) = -1;
                end
            end 
        end
    end
    % calculate alpha
    alpha(r) = 0.5*log((1-minerr)/minerr);
    % new weight
    for j = 1:total
        if minapply(j) == real(j)
            weight(j) = weight(j) * exp(-alpha(r));
        else
            weight(j) = weight(j)* exp(alpha(r));
        end
    end
    % noramlize new weight
    sumw = sum(weight);
    for j = 1:total
        weight(j) = weight(j)/sumw;
    end
end