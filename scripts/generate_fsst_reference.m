%% FSST Reference Data Generator
% Generates reference outputs from MATLAB's fsst() function for validation.
%
% Run this script in MATLAB, then copy the generated .mat file to the
% heart-sounds-segmentation project directory.
%
% Usage in MATLAB:
%   >> generate_fsst_reference
%
% Output:
%   fsst_reference.mat - Contains test signals and their FSST outputs

clear; clc;

fprintf('FSST Reference Data Generator\n');
fprintf('==============================\n\n');

%% Parameters (must match Python implementation)
fs = 1000;              % Sample rate (Hz)
window_len = 128;       % Window length
kaiser_beta = 0.5;      % Kaiser window beta parameter

% Create Kaiser window (same as scipy.signal.get_window(('kaiser', 0.5), 128, fftbins=False))
window = kaiser(window_len, kaiser_beta);

fprintf('Parameters:\n');
fprintf('  Sample rate: %d Hz\n', fs);
fprintf('  Window: Kaiser, length=%d, beta=%.1f\n', window_len, kaiser_beta);
fprintf('\n');

%% Test Case 1: Pure 100 Hz sinusoid
fprintf('Generating Test Case 1: 100 Hz sinusoid...\n');
duration1 = 0.5;  % 500 samples
t1 = (0:1/fs:duration1-1/fs)';
signal1 = sin(2*pi*100*t1);

[s1, f1, t1_out] = fsst(signal1, fs, window);

fprintf('  Signal length: %d samples\n', length(signal1));
fprintf('  Output spectrum size: %d x %d\n', size(s1, 1), size(s1, 2));
fprintf('  Frequency range: %.1f - %.1f Hz\n', f1(1), f1(end));

%% Test Case 2: Pure 50 Hz sinusoid
fprintf('Generating Test Case 2: 50 Hz sinusoid...\n');
duration2 = 0.5;
t2 = (0:1/fs:duration2-1/fs)';
signal2 = sin(2*pi*50*t2);

[s2, f2, t2_out] = fsst(signal2, fs, window);

fprintf('  Signal length: %d samples\n', length(signal2));
fprintf('  Output spectrum size: %d x %d\n', size(s2, 1), size(s2, 2));

%% Test Case 3: Linear chirp 50-200 Hz
fprintf('Generating Test Case 3: Linear chirp 50-200 Hz...\n');
duration3 = 1.0;  % 1000 samples
t3 = (0:1/fs:duration3-1/fs)';
f0 = 50; f1_chirp = 200;
signal3 = chirp(t3, f0, duration3, f1_chirp, 'linear');

[s3, f3, t3_out] = fsst(signal3, fs, window);

fprintf('  Signal length: %d samples\n', length(signal3));
fprintf('  Output spectrum size: %d x %d\n', size(s3, 1), size(s3, 2));

%% Test Case 4: Two-component signal (50 Hz + 150 Hz)
fprintf('Generating Test Case 4: Two-component signal (50 + 150 Hz)...\n');
duration4 = 0.5;
t4 = (0:1/fs:duration4-1/fs)';
signal4 = sin(2*pi*50*t4) + 0.5*sin(2*pi*150*t4);

[s4, f4, t4_out] = fsst(signal4, fs, window);

fprintf('  Signal length: %d samples\n', length(signal4));
fprintf('  Output spectrum size: %d x %d\n', size(s4, 1), size(s4, 2));

%% Test Case 5: DC signal
fprintf('Generating Test Case 5: DC signal...\n');
duration5 = 0.5;
signal5 = ones(duration5 * fs, 1);

[s5, f5, t5_out] = fsst(signal5, fs, window);

fprintf('  Signal length: %d samples\n', length(signal5));
fprintf('  Output spectrum size: %d x %d\n', size(s5, 1), size(s5, 2));

%% Test Case 6: Heart sound-like signal (realistic test)
fprintf('Generating Test Case 6: Simulated heart sound...\n');
duration6 = 2.0;  % 2000 samples (typical frame size)
t6 = (0:1/fs:duration6-1/fs)';
% Simulate S1 (~100-200 Hz) and S2 (~50-100 Hz) components
signal6 = 0.5*sin(2*pi*100*t6) .* exp(-5*mod(t6, 0.8)) + ...
          0.3*sin(2*pi*50*t6) .* exp(-5*mod(t6-0.3, 0.8)) + ...
          0.1*randn(size(t6));

[s6, f6, t6_out] = fsst(signal6, fs, window);

fprintf('  Signal length: %d samples\n', length(signal6));
fprintf('  Output spectrum size: %d x %d\n', size(s6, 1), size(s6, 2));

%% Save all reference data
fprintf('\nSaving reference data to fsst_reference.mat...\n');

reference = struct();

% Window
reference.window = window;
reference.fs = fs;
reference.window_len = window_len;
reference.kaiser_beta = kaiser_beta;

% Test case 1: 100 Hz sinusoid
reference.test1.name = '100 Hz sinusoid';
reference.test1.signal = signal1;
reference.test1.spectrum = s1;
reference.test1.frequencies = f1;
reference.test1.times = t1_out;

% Test case 2: 50 Hz sinusoid
reference.test2.name = '50 Hz sinusoid';
reference.test2.signal = signal2;
reference.test2.spectrum = s2;
reference.test2.frequencies = f2;
reference.test2.times = t2_out;

% Test case 3: Chirp
reference.test3.name = 'Linear chirp 50-200 Hz';
reference.test3.signal = signal3;
reference.test3.spectrum = s3;
reference.test3.frequencies = f3;
reference.test3.times = t3_out;

% Test case 4: Two-component
reference.test4.name = 'Two-component (50 + 150 Hz)';
reference.test4.signal = signal4;
reference.test4.spectrum = s4;
reference.test4.frequencies = f4;
reference.test4.times = t4_out;

% Test case 5: DC
reference.test5.name = 'DC signal';
reference.test5.signal = signal5;
reference.test5.spectrum = s5;
reference.test5.frequencies = f5;
reference.test5.times = t5_out;

% Test case 6: Heart sound
reference.test6.name = 'Simulated heart sound';
reference.test6.signal = signal6;
reference.test6.spectrum = s6;
reference.test6.frequencies = f6;
reference.test6.times = t6_out;

save('fsst_reference.mat', 'reference', '-v7.3');

fprintf('\nDone! File saved: fsst_reference.mat\n');
fprintf('\nTo validate, copy this file to the project and run:\n');
fprintf('  pixi run python scripts/compare_with_matlab.py\n');

%% Print summary statistics for quick verification
fprintf('\n');
fprintf('Quick verification values (compare with Python output):\n');
fprintf('=========================================================\n');

% Test 1: Peak frequency at middle time point
[~, peak_idx] = max(abs(s1(:, round(end/2))));
fprintf('Test 1 (100 Hz sine): Peak freq at t=0.25s: %.2f Hz\n', f1(peak_idx));

% Test 2: Peak frequency at middle time point
[~, peak_idx] = max(abs(s2(:, round(end/2))));
fprintf('Test 2 (50 Hz sine): Peak freq at t=0.25s: %.2f Hz\n', f2(peak_idx));

% Test 3: Peak frequencies at different times
for check_time = [0.2, 0.5, 0.8]
    t_idx = round(check_time * fs) + 1;
    [~, peak_idx] = max(abs(s3(:, t_idx)));
    expected_freq = f0 + (f1_chirp - f0) * check_time / duration3;
    fprintf('Test 3 (chirp): t=%.1fs, expected=%.1f Hz, got=%.2f Hz\n', ...
            check_time, expected_freq, f3(peak_idx));
end

% Test 5: DC magnitude
fprintf('Test 5 (DC): Magnitude at 0 Hz: %.4f\n', abs(s5(1, round(end/2))));

fprintf('\n');
