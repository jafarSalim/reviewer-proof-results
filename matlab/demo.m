%DEMO  The MATLAB templates in action: BPSK BER curve with exact 95 % intervals,
%   simulated until 100 errors per point, against theory. Writes ../figures/fig6_matlab_demo.png.
clear; close all; rng(3);
snr = 0:9;
ber = zeros(size(snr)); lo = ber; hi = ber; nb = ber;
for i = 1:numel(snr)
    ebn0 = 10^(snr(i)/10);
    blk = @() deal(sum((randn(2e5,1) + sqrt(2*ebn0)) < 0), 2e5);
    [ber(i), k, nb(i), lo(i), hi(i)] = ber_until_errors(blk, 100, 5e8);
    fprintf('Eb/N0 %d dB: BER %.3e  [%.3e, %.3e]  (%d errors in %.3g bits)\n', snr(i), ber(i), lo(i), hi(i), k, nb(i));
end
th = 0.5*erfc(sqrt(10.^(snr/10)));
h = figure('visible', 'off', 'position', [0 0 900 560]);
x = linspace(0, 9.3, 200);
semilogy(x, 0.5*erfc(sqrt(10.^(x/10))), '-', 'color', [0.5 0.5 0.55], 'linewidth', 1.5); hold on;
he = errorbar(snr, ber, ber - lo, hi - ber, 'o');
set(he, 'color', [0.10 0.62 0.42], 'markerfacecolor', [0.18 0.9 0.62], 'linewidth', 1.8, 'markersize', 7);
grid on; set(gca, 'fontsize', 13);
xlabel('E_b/N_0 [dB]'); ylabel('BER with exact 95% interval');
legend({'theory', 'simulated until 100 errors'}, 'location', 'southwest');
title('MATLAB / Octave template: ber\_until\_errors + ber\_ci');
inside = mean(th >= lo & th <= hi);
fprintf('theory inside the interval at %d of %d points\n', round(inside*numel(snr)), numel(snr));
set(h, 'PaperPositionMode', 'auto');
print(h, fullfile('..', 'figures', 'fig6_matlab_demo.png'), '-dpng', '-r170');
report_seeds([-14.2 -14.6 -13.9 -14.4 -14.1 -14.8 -14.0 -14.3]);
