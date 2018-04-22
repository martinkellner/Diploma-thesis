% Rozdeli skryte neurony na k klustrov podla receptivnych poli.
function clusterVisual(k)
    hidden = 100;       
    height = 48;
    width = 64;
    
    if (nargin==0) 
       k = 9; 
    end
    
    data = zeros(hidden, width*height);
    
    w = 640;
    h = 480;
    set(gcf, 'PaperUnits', 'points');
    set(gcf, 'PaperSize', [w*3 h*3]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition', [0 0 w*3 h*3]);    
    
    sides = ['l', 'r'];
    for s=1:2
        side = sides(s);
    
        format = strcat(side,'hid.%03d');

        for i=0:(hidden-1)          
            filename = sprintf(format,i);          
            one = load(filename);
            data(i+1,:) = one(:,3);                                  
        end; 

        opts = statset('Display','iter', 'MaxIter', 300);
        [idx, ctrs] = kmeans(data, k, ...
            'Distance', 'correlation',...                        
            'Options', opts);

        dlmwrite(strcat('clusters-visual/',side,'_idx'), idx);    
        dlmwrite(strcat('clusters-visual/',side,'_ctrs'), ctrs);
       
        r = round(sqrt(k));
        if (r*r < k)
            r = r+1;
        end;
        for i=1:k
           row = ctrs(i,:);              
           for h=1:height,
              b = (h-1)*width;
              z(1:width,h) = row(b+1:b+width);    
           end;
           subplot(r,r,i);
           contourf(1:width, 1:height, flipud(z'));             
        end;        
        print(strcat('clusters-visual/c_',side,'_clusters.png'),'-dpng');        
            
                        
        for i=1:hidden
           outname = sprintf('clusters-visual/%02d_%03d.png',idx(i),i);  
           gmFixedVisual(i, outname);
        end                
        
    end;
end






