% vykresli receptivne pole (64x48)
% vektor ma hodnoty v poradi ako keby sme citali receptive pole po riadkoch
function plotReceptiveField(field, width, height)
    
    if nargin==1
        width = 64;
        height = 48;
    elseif nargin ~= 3
        error('plotReceptiveField', 'Wrong number of input arguments')
    end
            
    z = zeros(width, height);    
    for ih=1:height,
       b = (ih-1)*width;       
       z(1:width,ih) = field(b+1:b+width);       
    end;                        
            
%     for ih=1:height
%         for iv=1:width
%             if z(iv, ih)>0
%                 M(iv,ih) = z(iv,ih);
%             end
%         end
%     end;
%     M = flipud(M');
%     
%     [rc,cc] = ndgrid(1:size(M,1),1:size(M,2));
%     Mt = sum(M(:));
%     c1 = sum(M(:) .* rc(:)) / Mt
%     c2 = sum(M(:) .* cc(:)) / Mt

    contourf(1:width, 1:height, flipud(z'));         
%     hold on;
%     scatter(c2, c1, 6*(width+height), 'ko', 'filled');    
%     hold off;
end