function image1 = showImage(path)
    image1 = double(squeeze(mean(imread(path),3)));
end