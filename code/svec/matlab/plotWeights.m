% Vykresli vahy ako stlpcovy graf.
function plotWeights(w, wlimit, color)    
    if nargin<2
       wlimit = 1;
    end
    if nargin<3
       color = 'b';
    end    
   
    bar(w, 'BaseValue', 0, 'FaceColor', color);
    
    ylim([-wlimit wlimit])
    xlim([0 length(w)+1])
    %plot(1:length(w), w, '-.bo', 'LineWidth', 2, 'MarkerFaceColor','r', 'MarkerSize', 10 );
     
    %axis([0 length(w)+1 -wlimit wlimit]);      
end