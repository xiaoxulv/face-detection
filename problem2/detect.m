function patchscore = detect(img1,scail)  
    path = 'MLSP_Images/Image1.jpg';
    img1 = double(squeeze(mean(imread(path),3)));
    witherr = 1;
    [E,mineigen,minthresh,alpha,sign] = adaBoost(witherr);
    
    %img1 = imresize(image1, scail*size(image));
    
    img1 = img1 - mean(img1(:));
    img1 = img1 / norm(img1(:));
    
    integral1 = cumsum(cumsum(img1,1),2);
    patchmeanofimg1 = zeros(size(img1,1)-N+1, size(img1,2)-M+1);
    for i = 1:size(img1,1)-N+1
        for j = 1:size(img1,2)-M+1
            a1 = integral1(i,j);
            a2 = integral1(i+N-1,j);
            a3 = integral1(i,j+M-1);
            a4 = integral1(i+N-1,j+M-1);
            patchmeanofimg1(i,j) = a4 + a1 - a2 - a3;
        end
    end
    
    W = zeros(1, size(E,2));
    for i = 1:size(E,2)
        W(i)= pinv(E(:,i))*patchmeanofimg1;
        
    end

    % convolution
%     tmpim = conv2(img1, fliplr(flipud(E)));
%     convolvedimage = tmpim(N:end, M:end);
% 
%     sumE = sum(E(:));
%     patchnew = imresize(patchmeanofimg1,size(convolvedimage));
%     patchscore = convolvedimage - sumE*patchnew(1:size(convolvedimage,1),1:size(convolvedimage,2));
end
    
    
    