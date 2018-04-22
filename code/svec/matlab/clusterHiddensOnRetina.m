function [idx, ctrs] = clusterHiddens(k)
    hidden = 100;
    
    left = zeros(hidden, 64*48);
    
    for i=0:(hidden-1)  
        % receptive field - left eye  
        filename = sprintf('lhid.%03d',i);  
        lhid = load(filename);
        left(i+1,:) = lhid(:,3);                  
    end; 
        
    [idx, ctrs] = kmeans(left, k);
    
    dlmwrite('k-left/idx', idx);
    dlmwrite('k-left/ctrs', ctrs);
    
    for i=1:k
       data = ctrs(i,:)';
       plotHiddenNeuron(data, [],[],[],[],[],[]);
       
       outname = sprintf('k-left/ctr_%02d.png',i);  
       print(outname,'-dpng');        
    end    

end






