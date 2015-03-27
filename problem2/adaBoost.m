
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


%threshold
mineigen = 0;
for i = 1:10
    maxV = max(W(i,:));
    minV = min(W(i,:));
    unit1 = (maxV-minV)/100;
    for k = 1:99
        thresh = minV + k*unit1;
        minthresh = thresh;
        apply = zeros(total,1);% predict classification for all
        for j = 1:total
            if W(1,j) > thresh
                apply(j) = 1;
            end
        end
        % error weight sum
        err = 0;
        minerr = 1;
        for j = 1:total
            if apply(j) ~= real(j)
                err = err + weight(j);
            end
        end
        if(err < minerr)
            minerr = err;
            minthresh = thresh;
            mineigen = i;
        end
    end
end
