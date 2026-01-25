% generate_wsst_reference.m
% Generates reference outputs from MATLAB's wsst() function for validation.
%
% This script creates test signals and computes their WSST using MATLAB's
% built-in wsst() function, then saves the results for comparison.
%
% Usage:
%   >> generate_wsst_reference
%
% Output:
%   wsst_reference.mat - Contains test signals and their WSST outputs

%% Parameters
fs = 1000;  % Sample rate (Hz)
num_voices = 32;  % Voices per octave (MATLAB default)

%% Test 1: 100 Hz sinusoid
t1 = (0:999)' / fs;
signal1 = sin(2*pi*100*t1);
[s1, f1, t1_out] = wsst(signal1, fs, 'amor', 'VoicesPerOctave', num_voices);

reference.test1.name = '100 Hz sinusoid';
reference.test1.signal = signal1;
reference.test1.spectrum = s1;
reference.test1.frequencies = f1;
reference.test1.times = t1_out;

%% Test 2: 50 Hz sinusoid
t2 = (0:499)' / fs;
signal2 = sin(2*pi*50*t2);
[s2, f2, t2_out] = wsst(signal2, fs, 'amor', 'VoicesPerOctave', num_voices);

reference.test2.name = '50 Hz sinusoid';
reference.test2.signal = signal2;
reference.test2.spectrum = s2;
reference.test2.frequencies = f2;
reference.test2.times = t2_out;

%% Test 3: Linear chirp 50-200 Hz
t3 = (0:1999)' / fs;
signal3 = chirp(t3, 50, t3(end), 200);
[s3, f3, t3_out] = wsst(signal3, fs, 'amor', 'VoicesPerOctave', num_voices);

reference.test3.name = 'Linear chirp 50-200 Hz';
reference.test3.signal = signal3;
reference.test3.spectrum = s3;
reference.test3.frequencies = f3;
reference.test3.times = t3_out;

%% Test 4: Two-component signal (50 Hz + 150 Hz)
t4 = (0:999)' / fs;
signal4 = sin(2*pi*50*t4) + 0.5*sin(2*pi*150*t4);
[s4, f4, t4_out] = wsst(signal4, fs, 'amor', 'VoicesPerOctave', num_voices);

reference.test4.name = 'Two-component (50 + 150 Hz)';
reference.test4.signal = signal4;
reference.test4.spectrum = s4;
reference.test4.frequencies = f4;
reference.test4.times = t4_out;

%% Save parameters
reference.fs = fs;
reference.num_voices = num_voices;
reference.wavelet = 'amor';

%% Save to file
fprintf('\nSaving reference data to wsst_reference.mat...\n');

% Display summary
fprintf('\nReference data summary:\n');
fprintf('  Sample rate: %d Hz\n', fs);
fprintf('  Wavelet: amor (analytic Morlet)\n');
fprintf('  Voices per octave: %d\n', num_voices);
fprintf('\n');

test_names = {'test1', 'test2', 'test3', 'test4'};
for i = 1:length(test_names)
    test = reference.(test_names{i});
    fprintf('  %s: %s\n', test_names{i}, test.name);
    fprintf('    Signal length: %d samples\n', length(test.signal));
    fprintf('    Spectrum size: %d x %d (freq x time)\n', size(test.spectrum, 1), size(test.spectrum, 2));
    fprintf('    Frequency range: %.1f - %.1f Hz\n', min(test.frequencies), max(test.frequencies));
end

save('wsst_reference.mat', 'reference', '-v7.3');

fprintf('\nDone! File saved: wsst_reference.mat\n');
fprintf('Copy this file to the scripts/ directory and run compare_wsst_with_matlab.py\n');
