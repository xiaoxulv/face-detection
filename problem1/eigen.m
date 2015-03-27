function eigenface = eigen(location)
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
    % get eigen face
    [U,S,V] = svd(Y,0);
    eigenfacevector = U(:,1);
    [nrows, ncolumns] = size(images{1});
    eigenface = reshape(eigenfacevector, nrows, ncolumns);
    % imagesc(eigenface)
end