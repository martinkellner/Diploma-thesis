% Nacita obrazok pre lave a prave oko do vektora.
function [left, right] = loadVisualInput(imgNum, height)

    if nargin==1        
        height = 48;
    elseif nargin<3
        error('loadImages, args: %d, %d', 'Wrong number of input parameters')
    end

    if isscalar(imgNum)
        filename = sprintf('img%03d',imgNum);      
        imgs = load(filename);

        left = imgs(1:height, :)';
        left = left(:);

        right = imgs(height+1:end, :)';
        right = right(:);
    else
        for i=1:length(imgNum)
            filename = sprintf('img%03d',imgNum(i));      
            imgs = load(filename);

            tmp = imgs(1:height, :)';
            left(i,:) = tmp(:);

            tmp = imgs(height+1:end, :)';
            right(i,:) = tmp(:);
        end
    end

    
    
    

end