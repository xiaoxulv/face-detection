function [F,facesize] = face(location) 
    fileNames = dir(fullfile(location, '*.pgm'));
    len = length(fileNames);
    images = cell(len,1);
    for i = 1:len
        path = strcat(location, fileNames(i).name);
        x = double(histeq(imread(path)));
        x = x - mean(x(:));
        if norm(x(:)) ~= 0
            x = x / norm(x(:));% variance normalization
        end
        images{i} = x;
        facesize = size(x,1);
    end
    F = [];
    for i = 1:len
        F = [F images{i}(:)];
    end
end
