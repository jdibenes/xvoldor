
mag_step = 2;
max_mag = 50;
enable_plots = false;

alpha_all = [];
beta_all = [];

for mag_base = 0:mag_step:max_mag
select = (mag >= mag_base) & (mag < (mag_base + mag_step));
res_sub = res(select);
res_sub = res_sub.^2;
res_sub_s = sort(res_sub, 'ascend');

max_p = res_sub_s(ceil(0.75*numel(res_sub_s)));
width = max_p / 1000;
edges = 0:width:max_p;
C = histcounts(res_sub_s, edges);
P = C / (numel(res_sub_s) * width);
X = edges(1:(end-1));

alpha_direct = median(res_sub_s);

idx_step = 1;
idx = 1:idx_step:numel(res_sub_s);
p = (idx - 1) / (numel(res_sub_s) - 1);
keep = (p > 0.1) & (p < 0.9) & ~((p > 0.4) & (p < 0.6));
idx = idx(keep);
p = p(keep);
q = res_sub_s(idx);
beta_direct = median(log(p(:) ./ (1 - p(:))) ./ log(q(:) ./ alpha_direct));

F_direct = fisk(edges(2:end) - (width/2), alpha_direct, beta_direct);

if (enable_plots)
    figure()
    plot(X, P)
    hold on
    plot(X, F_direct);
    title(['mag ' num2str(mag_base) '-' num2str(mag_base + mag_step)])
end

alpha_all = [alpha_all; alpha_direct];
beta_all = [beta_all; beta_direct];
end

FX = (0:mag_step:max_mag) + (mag_step / 2);

beta_params  = [FX(:), ones(numel(FX), 1)] \ beta_all;
alpha_params = [FX(:), ones(numel(FX), 1)] \ log(alpha_all);

figure()
plot(FX, alpha_all)
hold on
plot(FX, exp(alpha_params(1)*FX + alpha_params(2)))
set(gca, 'YScale', 'log')
title('alpha');

figure()
plot(FX, beta_all);
hold on
plot(FX, beta_params(1)*FX + beta_params(2))
title('beta');








%{
max_p = res_sub_s(ceil(0.75*numel(res_sub_s)));
width = max_p / 1000;
edges = 0:width:max_p;
%width = 1e-2; % choose?
%edges = 0:width:1; % choose max?

C = histcounts(res_sub_s, edges);
P = C / (numel(res_sub_s) * width);
X = edges(1:(end-1));

plot(X, P)
hold on
drawnow





cdf = cumsum(P*width);
median_index = find((cdf >= 0.5), 1);
alpha_init = edges(median_index);






quart_index = find((cdf >= 0.25), 1);
cdf_quart = cdf(quart_index);
beta_init = log(cdf_quart/(1-cdf_quart)) / log(edges(quart_index)/alpha_init);

threequarts_index = find((cdf >= 0.66), 1);
cdf_threequarts = cdf(threequarts_index);
beta_init_2 = log(cdf_threequarts/(1-cdf_threequarts)) / log(edges(threequarts_index)/alpha_init);

F = fisk(edges(2:end) - (width/2), alpha_init, beta_init_2);
plot(X, F);


legend({'base', 'init', 'direct'})
%}

%alpha_init = P(find((cumsum(P) >= 0.5), 1));

%h_opt = @(v)(P - fisk(edges(2:end), v(1), v(2)));
%x_sol = lsqnonlin(h_opt, [alpha_init, beta_init]);

%G = fisk(edges(2:end), x_sol(1), x_sol(2));
%plot(X, G);

%alpha_all = [alpha_all; real(x_sol(1))];
%beta_all = [beta_all; real(x_sol(2))];

%alpha_all_init = [alpha_all_init; alpha_init];
%beta_all_init = [beta_all_init; beta_init_2];


%{
select = mag >= 25 & mag <= 30;
res_sub = res(select);
res_sub = res_sub.^2;
res_sub_s = sort(res_sub, 'ascend');

width = 1e-2;
edges = 0:width:1;
C = histcounts(res_sub_s, edges);
P = C / (numel(res_sub_s) * width);

beta=0.9;
alpha=median(P);

F = fisk(edges(2:end), alpha, beta);
X = edges(1:(end-1));

plot(X, P)
hold on
%plot(X, F)
%histogram(res_sub_s, 0:1e-3:1);

h_opt = @(v)(P - fisk(edges(2:end), v(1), v(2)));

x_sol = lsqnonlin(h_opt, [1, 1]);

x_sol
G = fisk(edges(2:end), x_sol(1), x_sol(2));

plot(X, G);
%}

function f = fisk(x, a, b)
f = ((b./a).*((x./a).^(b-1)))./((1+((x./a).^b)).^2);
end