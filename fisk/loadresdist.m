
clear all

model = 'ptl-neuflow2';
data = load(['./residuals/sintel/residuals_' model]);

res = data.res;
mag = data.mag;
