function [lo, hi] = ber_ci(k, n, conf)
%BER_CI  Exact (Clopper-Pearson) confidence interval for an error rate k/n.
%   [lo, hi] = ber_ci(k, n)        95 % interval
%   [lo, hi] = ber_ci(k, n, 0.99)  99 % interval
%   Uses betaincinv, which is in base MATLAB and in GNU Octave (no toolbox).
%   With k = 0 the upper limit is close to 3/n (the "rule of three").
if nargin < 3, conf = 0.95; end
a = 1 - conf;
lo = zeros(size(k));  hi = ones(size(k));
m = k > 0;
lo(m) = betaincinv(a/2, k(m), n(m) - k(m) + 1);
m = k < n;
hi(m) = betaincinv(1 - a/2, k(m) + 1, n(m) - k(m));
end
