
clear all

model = 'searaft';
data = load(['./residuals/residuals_' model]);

res = data.res;
mag = data.mag;
