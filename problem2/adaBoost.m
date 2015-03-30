function [E,mineigen,minthresh,alpha,sign]= adaBoost(witherr)
    facelocation = 'BoostingData/train/face/';
    [F,facesize] = face(facelocation);
    nonfacelocation = 'BoostingData/train/non-face/';
    [NF,~] = face(nonfacelocation);

    location = 'lfw1000/';
    E = eigens(location,facesize);
    % !!!!!!!!!!!!!!!!!!!!Wrong!!!!!!!!!!!!!!!!!
    % weight of eigenfaces in face and non-face
    % Wf = pinv(E)*F;
    % Wnf = pinv(E)*NF;
    % W = [Wf Wnf];
    Wf = zeros(size(E,2), size(F,2));
    Wnf = zeros(size(E,2), size(NF,2));
    for i = 1:size(E,2)
        Wf(i,:) = pinv(E(:,i))*F;
        Wnf(i,:) = pinv(E(:,i))*NF;
        F = F - E(:,i)*Wf(i,:);
        NF = NF - E(:,i)*Wnf(i,:);
    end
    W = [Wf Wnf];

    if witherr == 1
        ef = sum(F.^2)/size(F,1);
        enf = sum(NF.^2)/size(NF,1);
        e = [ef enf];
        W = [W;e];
    end

    fnum = size(Wf,2);
    nnum = size(Wnf,2);
    total = size(W,2);
    wunit = 1/total;
    weight = repmat(wunit,total,1);% initial weight for all
    ff = ones(fnum,1);
    nn = zeros(nnum,1);
    real = [ff;nn];% real classification for all


    % adaboost loop, rounds
    round = 20;
    alpha = zeros(round,1);
    mineigen = zeros(round,1);
    minthresh = 100;
    minthresh = repmat(minthresh,round,1);
    sign = ones(round,1); 

    for r = 1:round
        minerr = 100;
        minapply = zeros(total,1);
        for i = 1:size(W,1) % for 20 eigenfaces
            maxV = max(W(i,:));
            minV = min(W(i,:));
            unit1 = (maxV-minV)/100;
            for k = 1:100 % for 100 gaps
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
                    apply = apply~=1;
                end
                % update minimum error
                if err < minerr
                    minerr = err;
                    minthresh(r) = thresh;
                    mineigen(r) = i;
                    minapply = apply;
                    if b == 1
                        sign(r) = -1;
                    else
                        sign(r) = 1;
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
end



