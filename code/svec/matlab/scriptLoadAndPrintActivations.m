[activations] = loadActivations();
figure;
plotActivationsAsGlyphs(activations);
print('activations/activations.png','-dpng')