
facelocation = 'BoostingData/train/face/';
[F,facesize] = face(facelocation);
nonfacelocation = 'BoostingData/train/non-face/';
[NF,nfsize] = face(nonfacelocation);

location = 'lfw1000/';
E = eigens(location,facesize);
% weight of eigenfaces in face and non-face
Wf = pinv(E)*F;
Wnf = pinv(E)*NF;
W = [Wf Wnf];

fnum = size(Wf,2);
nnum = size(Wnf,2);
total = size(W,2);
wunit = 1/total;
weight = repmat(wunit,total,1);% initial weight for all
ff = ones(fnum,1);
nn = zeros(nnum,1);
real = [ff;nn];% real classification for all


% adaboost loop, rounds
round = 5;
alpha = zeros(round,1);
mineigen = zeros(round,1);
minthresh = 100;
minthresh = repmat(minthresh,round,1);

for r = 1:round
    minerr = 100;
    minapply = zeros(total,1);
    for i = 1:10 % for 10 eigenfaces
        maxV = max(W(i,:));
        minV = min(W(i,:));
        unit1 = (maxV-minV)/100;
        for k = 1:99 % for 100 gaps
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
            % update minimum error
            if(err < minerr)
                minerr = err;
                minthresh(r) = thresh;
                mineigen(r) = i;
                minapply = apply;
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



