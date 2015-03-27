function E = eigens(location,facesize)
    % first read all the faces
    fileNames = dir(fullfile(location, '*.pgm'));
    len = length(fileNames);
    images = cell(len,1);
    for i = 1:len
        path = strcat(location, fileNames(i).name);
        x = double(imread(path));
        x = x - mean(x(:));% mean normalization
        x = x / norm(x(:));% variance normalization
        images{i} = x;
    end
    % compose matrix from faces collection
    Y = [];
    for i = 1:len
        Y = [Y images{i}(:)];
    end

    [U,~,~] = svd(Y,0);
    eigenfaces = cell(10,1);
    for i = 1:10
        eigenfacevectors = U(:,i);
        %[nrows, ncolumns] = size(images{i});
        eigenfacevectors=reshape(eigenfacevectors,[64 64]);
        t = imresize(eigenfacevectors, [facesize, facesize]);% resize here
        %eigenfaces{i} = reshape(t, facesize*facesize,1);
        eigenfaces{i}=t(:);
    end
    
    E = [];
    for i = 1:10
        E = [E eigenfaces{i}(:)];
    end
end