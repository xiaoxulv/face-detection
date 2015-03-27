function patchscore = detect(image1, scail)
    location = 'lfw1000/';
    E = eigen(location);
    [N, M] = size(E);

    %scail here
    img1 = imresize(image1, scail*size(image1));
    
    img1 = img1 - mean(img1(:));% mean normalization
    img1 = img1 / norm(img1(:));% variance normalization
    % integral image trick from Viola-Jones
    integral1 = cumsum(cumsum(img1,1),2);% cumulative sum
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

    % convolution
    tmpim = conv2(img1, fliplr(flipud(E)));
    convolvedimage = tmpim(N:end, M:end);

    sumE = sum(E(:));
    patchnew = imresize(patchmeanofimg1,size(convolvedimage));
    patchscore = convolvedimage - sumE*patchnew(1:size(convolvedimage,1),1:size(convolvedimage,2));
    
end