% Nacita i-ty skryty neuron, vrati [lhid, rhid, thid, vhid]
% Ak nie je specifikovany onlyData = 0, tak sa vratia iba vahy, 
% inak vsetky udaje
function [lhid, rhid, thid, vhid] = loadHiddenWeights(i, onlyData)
    if (nargin == 1)
        onlyData = 1;
    end

    % receptive field - left eye  
    filename = sprintf('lhid.%03d',i);  
    lhid = load(filename);    
          
    % right eye
    filename = sprintf('rhid.%03d',i);  
    rhid = load(filename);

    % tilt
    filename = sprintf('thid.%03d',i);  
    thid = load(filename);
    
    % version
    filename = sprintf('vhid.%03d',i);  
    vhid = load(filename);            

    if (onlyData == 1)
       lhid = lhid(:,3);
       rhid = rhid(:,3);
       thid = thid(:,2);
       vhid = vhid(:,2);
    end
end






