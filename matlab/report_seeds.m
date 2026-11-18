function s = report_seeds(x, conf)
%REPORT_SEEDS  Mean, standard deviation and t-based confidence interval of the mean over
%   repeated runs (e.g. training seeds). Prints a line you can paste into a paper.
if nargin < 2, conf = 0.95; end
x = x(:);  n = numel(x);
s.mean = mean(x);  s.std = std(x);  s.n = n;
% t quantile by bisection: for t > 0, CDF = 1 - betainc(nu/(nu+t^2), nu/2, 1/2)/2
nu = n - 1;  p = 1 - (1 - conf)/2;  lo = 0;  hi = 100;
for it = 1:200
    t = (lo + hi)/2;
    cdf = 1 - 0.5*betainc(nu/(nu + t^2), nu/2, 0.5);
    if cdf < p, lo = t; else, hi = t; end
end
s.half = t * s.std / sqrt(n);
fprintf('%.3f +- %.3f (mean +- %g%% CI half-width over %d runs; std %.3f; range %.3f to %.3f)\n', ...
        s.mean, s.half, 100*conf, n, s.std, min(x), max(x));
end
