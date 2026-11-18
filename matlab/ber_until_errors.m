function [ber, k, n, lo, hi] = ber_until_errors(sim_block, target_errors, max_bits)
%BER_UNTIL_ERRORS  Run a Monte Carlo link simulation until enough errors are counted.
%   sim_block: function handle, [errors, bits] = sim_block() simulates one block.
%   target_errors: stop after this many errors (100 gives a +-20 % 95 % interval).
%   max_bits: hard limit; if reached, the result is an upper bound only.
%
%   Example (BPSK in AWGN at 6 dB):
%     ebn0 = 10^(6/10);
%     blk = @() deal(sum((randn(1e5,1) + sqrt(2*ebn0)) < 0), 1e5);
%     [ber, k, n, lo, hi] = ber_until_errors(blk, 100, 1e9)
if nargin < 2, target_errors = 100; end
if nargin < 3, max_bits = 1e9; end
k = 0;  n = 0;
while k < target_errors && n < max_bits
    [e, b] = sim_block();
    k = k + e;  n = n + b;
end
ber = k / n;
[lo, hi] = ber_ci(k, n);
if k < target_errors
    warning('ber_until_errors:limit', 'stopped at %g bits with only %d errors', n, k);
end
end
