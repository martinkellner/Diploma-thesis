% vrati tazisko 2D matice
function [c1, c2] = centerOfMass(M)
   [rc,cc] = ndgrid(1:size(M,1),1:size(M,2));
    Mt = sum(M(:));
    c1 = sum(M(:) .* rc(:)) / Mt;
    c2 = sum(M(:) .* cc(:)) / Mt;
end