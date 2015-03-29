% location = 'lfw1000/';
% E = eigens(location,19);

% train to get strong classifier from weak classifiers
[E,mineigen,minthresh,alpha,sign] = adaBoost();

[testface, ~] = face('BoostingData/test/face/');
[testnonface,~] = face('BoostingData/test/non-face/');
% Wf = pinv(E)*testface;
% Wnf = pinv(E)*testnonface;
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

num = size(W,2);
val = zeros(num,1);
res = zeros(num,1);

real = [ones(size(Wf,2),1);zeros(size(Wnf,2),1)];

for i = 1:size(mineigen,1)
    e = mineigen(i);
    a = alpha(i); 
    for j = 1:num
        if sign(i) == 1
            if W(e,j) > minthresh(i)
                val(j) = val(j) + a;
            else
                val(j) = val(j) - a;
            end
        else
            if W(e,j) < minthresh(i)
                val(j) = val(j) + a;
            else
                val(j) = val(j) - a;
            end
        end
    end
end

for i = 1:num
    if val(i) > -1.3 % shift here...
        res(i) = 1;
    else
        res(i) = 0;
    end
end
count1 = 0;
count2 = 0;
for i = 1:size(Wf,2)
    if res(i) == real(i)
        count1 = count1 + 1;
    end
end
rate1 = count1/size(Wf,2);
for i = size(Wf,2)+1:size(Wnf,2)
    if res(i) == real(i)
        count2 = count2 + 1;
    end
end
rate2 = count2/size(Wnf,2);
rate = (count1+count2)/size(W,2);

beep;
        