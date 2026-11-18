# Results checklist for simulation and machine-learning papers

Each item lists the reviewer question it answers and where this repository shows why it matters.

## Monte Carlo error rates

1. **At least 100 errors per point.** *"How many errors were counted at the highest SNR?"* With k errors the 95 % interval is roughly ±1.96/√k of the estimate: ±20 % at 100 errors, ±62 % at 10, a factor of several below 5. (Experiment 1.)
2. **No zero-error points drawn as values.** Zero errors in n bits means BER < 3/n at 95 % (rule of three). Draw it as an upper bound or leave it out.
3. **Intervals on the curves.** Clopper–Pearson (exact, conservative) or Wilson (close to nominal). Never the Wald interval with few errors: its real coverage drops to 63–80 %.
4. **State the stopping rule** ("each point was simulated until 100 errors or 10⁹ bits") and the random-number generator and seed.

## Data and splits

5. **Split by the unit that generates the data**: channel realization, user, location, session, patient, recording. Report the overlap between training and test sets (it should be 0). (Experiment 2.)
6. **Hold out the test set before any tuning.** Hyperparameters chosen on the test set make it a validation set.
7. **Generate the test channels with a different seed** from the training channels, and say so.
8. **Sanity bound:** for linear Gaussian models the MMSE/LMMSE estimator is optimal. A learned estimator that beats it on such data has leaked.

## Baselines

9. **The strongest classical method**, with honest assumptions (LMMSE with estimated statistics, ML detection where feasible), not only LS or zero-forcing. (Experiment 3.)
10. **The full operating range.** Train at one SNR and test across the range, or train across the range; show both ends.
11. **Complexity next to accuracy:** multiplications per symbol, parameters, run time. A 0.3 dB gain at 100× the cost is a different claim.
12. **Tune the baselines as carefully as the proposal** (same budget of trials), and report the settings.

## Randomness in training

13. **At least 5 seeds**, mean ± confidence interval, the same seeds for every method. (Experiment 4.)
14. **Never report the best run** as the result. If you must, report it as "best of k" and give k.
15. **Report the spread** (min–max or standard deviation) so readers can judge whether a gain is inside it.

## Significance

16. **Match the test to the design**: McNemar on one shared test set; 5×2cv paired t-test or the corrected resampled t-test for repeated splits. Avoid the plain paired t-test on cross-validation folds. (Experiment 5.)
17. **Enough test samples to detect the effect you claim**: about 5,000 for a 1 % accuracy gain at 80 % power when the models disagree on a few percent of samples.
18. **Correct for multiple comparisons** (Holm–Bonferroni) when you test many metrics, datasets or SNRs.
19. **Effect size with its interval**, not only a p-value.

## Reproducibility

20. **Publish the code, configuration, seeds and the exact data split**, and a single command that regenerates every figure.
21. **Pin library versions** (requirements file, MATLAB release).
22. **Say what was run where**: if figures were produced in GNU Octave rather than MATLAB, or on a GPU with nondeterministic kernels, state it.
