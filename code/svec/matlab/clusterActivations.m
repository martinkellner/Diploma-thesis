function clusterActivations(k)
    if (nargin==0)
        k = 12;
    end

    [activations, ~, ~, ~, ~, ~, ~] = loadActivations();

    data = activations';
            
    opts = statset('Display','iter', 'MaxIter', 300);
    [idx, ctrs] = kmeans(data, k, ...            
            'Options', opts);
    
    dlmwrite('clusters-activations/clusters_activations.idx', idx);    
    dlmwrite('clusters-activations/clusters_activations.ctrs', ctrs); 
    
    figure    
    plotActivationsAsGlyphs(ctrs);
    print('clusters-activations/ctrs_all','-dpng');
    
    figure
    
    w = 640;
    h = 480;
    set(gcf, 'PaperUnits', 'points');
    set(gcf, 'PaperSize', [w*3 h*3]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition', [0 0 w*3 h*3]);
    
    for i=1:k
        plotActivationsAsBubbles(ctrs(i,:));                        
        outname = sprintf('clusters-activations/ctrs_%03d.png',i);   
        print(outname,'-dpng');                 
    end
    

end