function location = maxLocation(patchscore, scail)
    hLocalMax = vision.LocalMaximaFinder;
    hLocalMax.MaximumNumLocalMaxima = 12;
    hLocalMax.NeighborhoodSize = [199 199];
    hLocalMax.Threshold = mean(patchscore(:))+ std(patchscore(:));
    % hLocalMax.Threshold = 1;
    location = step(hLocalMax, patchscore)/scail;
   
end
    