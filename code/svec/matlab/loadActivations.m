% nacita data zo suborov - [activations, choosen, tilts, versions, factorGaze, factorVisual, hbiases]
function [activations, choosen, tilts, versions, factorGaze, factorVisual, hbiases] = loadActivations(i)
    
    choosen = load('choosen');
    activations = load('pall_activations');
    tilts = load('tilts');
    versions = load('versions');    
    
    factors = load('factors');
    factorGaze = factors(1);
    factorVisual = factors(2);
    
    hbiases = load('hbiases');
    hbiases = hbiases(:,1);
        
end






