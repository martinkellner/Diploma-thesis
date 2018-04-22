
    
        w = 640;
        h = 480;
        set(gcf, 'PaperUnits', 'points');
        set(gcf, 'PaperSize', [w*3 h*1]);
        set(gcf, 'PaperPositionMode', 'manual');
        set(gcf, 'PaperPosition', [0 0 w*3 h*1]);

    for i=0:100


        [lhid, rhid] = loadHiddenWeights(i);
        subplot(1,3,1)
        plotReceptiveField(lhid);
        subplot(1,3,2)
        plotReceptiveField(lhid);
        subplot(1,3,3)
        plotReceptiveField((lhid+rhid)/2);


         outname = sprintf('lala/lala_%03d.png',i+1);      
         print(outname,'-dpng');  
    end

