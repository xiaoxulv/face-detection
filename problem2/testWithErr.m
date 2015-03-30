location = 'lfw1000/';
E = eigens(location,19);

% train to get strong classifier from weak classifiers
%[E,mineigen,minthresh,alpha] = adaBoost();

[testface, ~] = face('BoostingData/test/face/');
[testnonface,~] = face('BoostingData/test/non-face/');

Wf = zeros(size(E,2), size(testface,2));
Wnf = zeros(size(E,2), size(testnonface,2));
originNonFace = testnonface;
originFace = testface;
for i = 1:size(E,2)
    Wf(i,:) = pinv(E(:,i)) * testface;
    Wnf(i,:) = pinv(E(:,i)) * testnonface;
    testface = testface - E(:,i)*Wf(i,:);
    testnonface = testnonface - E(:,i)*Wnf(i,:);
end


ef = sum(testface.^2)/size(testface,1);
enf = sum(testnonface.^2)/size(testnonface,1);


num = size(W,2);
val = zeros(num,1);
res = zeros(num,1);

real = [ones(size(Wf,2),1);zeros(size(Wnf,2),1)];

for i = 1:size(mineigen,1)
    e = mineigen(i);
    a = alpha(i); 
    for j = 1:num
        if W(i,j) > minthresh(i)
            val(j) = val(j) + a;
        else
            val(j) = val(j) - a;
        end
    end
end

for i = 1:num
    if res(i) > 0
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

beep;
        