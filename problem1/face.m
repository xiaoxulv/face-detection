
scail = [0.5 0.75 1 1.5 2];
path1 = 'MLSP_Images/Image4.jpg';
% path1 = 'testEigen.jpg';
image1 = showImage(path1);
imagesc(image1);
for i = 1:length(scail)
    patchscore = detect(image1, scail(i));
    location = maxLocation(patchscore, scail(i));
    for j = 1:size(location,1)
        rectangle('Position',[location(j,:),64,64],'LineWidth',2,'EdgeColor','r');
    end
end







