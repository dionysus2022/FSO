% rx_exp_cdm_dataset_generator_sc1.m
% Extract one blind Distorted CDM sample from each received OFDM subcarrier.
% All 123 active subcarriers are retained, so each frame produces 123 samples.
% Frame IDs 1:60 are training frames and 61:75 are validation frames.
% This script uses a dedicated sc1 directory and never overwrites sc5 results.

clear; clear global; close all; clc;

%% Paths
project_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master';
addpath(genpath(project_root));
global PROG; PROG.showMessagesLevel = 2; initProg();

data_root = fullfile(project_root, '2_Data_Results');
rx_base = fullfile(data_root, 'rx_data', '2026.06.21');
dataset_root = fullfile(data_root, 'dataset_cdm_sc1_single', 'dataset');
train_root = fullfile(dataset_root, 'train');
val_root = fullfile(dataset_root, 'val');
if ~exist(train_root, 'dir')
    mkdir(train_root);
end
if ~exist(val_root, 'dir')
    mkdir(val_root);
end

%% Dataset configuration
mod_list = {'2QAM', '4QAM', '16QAM', '64QAM', '256QAM'};
mod_bits = [1, 2, 4, 6, 8];
sub_list = {'01', '02', '03'};
frames_per_sub = 25;
scope_Fs = 80e9;
cdm_size = 64;
train_frame_ids = 1:60;
val_frame_ids = 61:75;
total_active_subcarriers = 123;

%% OFDM receiver configuration
SIG.nSyms = 2^7;
ofdm.NumberOfIFFTSamples = 256;
ofdm.Carrier_location = 4:126;
ofdm.Carrier_location_demo = [4:126, 132:254];
ofdm.NumberOfCarriers = length(ofdm.Carrier_location);
ofdm.NumberOfCarriers_demo = length(ofdm.Carrier_location_demo);
ofdm.NumberOfGuardTime = 16;
ofdm.size = SIG.nSyms;

if ofdm.NumberOfCarriers ~= total_active_subcarriers
    error('Expected %d active subcarriers, but receiver defines %d.', ...
        total_active_subcarriers, ofdm.NumberOfCarriers);
end

total_saved_count = 0;
train_saved_count = 0;
val_saved_count = 0;
total_skipped_count = 0;

fprintf('\nExtracting blind CDM features from independent subcarriers...\n');
fprintf('Output: %s\n', dataset_root);

%% Process all received signals
for mod_idx = 1:numel(mod_list)
    modulation = mod_list{mod_idx};
    Label_Bits = mod_bits(mod_idx);

    fprintf('\n=== %s ===\n', modulation);

    for sub_idx = 1:numel(sub_list)
        sub_name = sub_list{sub_idx};
        rx_dir = fullfile(rx_base, modulation, ['sub' sub_name]);

        for local_idx = 1:frames_per_sub
            signal_id = (sub_idx - 1) * frames_per_sub + local_idx;
            global_t_pos = signal_id;
            bin_file = fullfile(rx_dir, sprintf('%d.bin', signal_id));

            if ismember(global_t_pos, train_frame_ids)
                Split = 'train';
                split_class_dir = fullfile(train_root, modulation);
            elseif ismember(global_t_pos, val_frame_ids)
                Split = 'val';
                split_class_dir = fullfile(val_root, modulation);
            else
                fprintf('  [Skipped] Frame %d is not assigned to a split.\n', ...
                    global_t_pos);
                total_skipped_count = total_skipped_count + 1;
                continue;
            end
            if ~exist(split_class_dir, 'dir')
                mkdir(split_class_dir);
            end

            if ~exist(bin_file, 'file')
                fprintf('  [Skipped] Missing %s\n', bin_file);
                total_skipped_count = total_skipped_count + 1;
                continue;
            end

            file_info = dir(bin_file);
            if file_info.bytes == 0
                fprintf('  [Skipped] Empty %s\n', bin_file);
                total_skipped_count = total_skipped_count + 1;
                continue;
            end

            try
                OutputFSO = readKeysightBin(bin_file);
                OutputFSO = resample(OutputFSO, 16e9, scope_Fs);
                OutputFSO = OutputFSO - mean(OutputFSO);

                mean_amplitude = mean(abs(OutputFSO));
                if ~isfinite(mean_amplitude) || mean_amplitude <= eps
                    error('Received signal has invalid amplitude.');
                end
                data_in = OutputFSO / mean_amplitude;

                rx_serial = deOFDM(data_in, ofdm, SIG.nSyms);
                expected_symbols = SIG.nSyms * ofdm.NumberOfCarriers_demo;
                if numel(rx_serial) < expected_symbols
                    error('deOFDM returned %d symbols; expected at least %d.', ...
                        numel(rx_serial), expected_symbols);
                end

                rx_matrix = reshape(rx_serial(1:expected_symbols), ...
                    SIG.nSyms, ofdm.NumberOfCarriers_demo).';
                rx_block = rx_matrix(1:ofdm.NumberOfCarriers, :);

                for subcarrier_idx = 1:total_active_subcarriers
                    rx_syms = rx_block(subcarrier_idx, :).';

                    rms_value = sqrt(mean(abs(rx_syms).^2));
                    if ~isfinite(rms_value) || rms_value <= eps
                        error('Subcarrier %d has invalid RMS.', subcarrier_idx);
                    end
                    rx_syms = rx_syms / rms_value;

                    Distorted_CDM = generate_CDM_Smooth(rx_syms, cdm_size);
                    Frame_ID = global_t_pos;
                    Subcarrier_ID = subcarrier_idx;
                    FFT_Bin = ofdm.Carrier_location(subcarrier_idx);
                    Symbols_Per_Sample = numel(rx_syms);
                    Source_Subfolder = ['sub' sub_name];
                    Source_File = sprintf('%d.bin', signal_id);

                    save_name = fullfile(split_class_dir, ...
                        sprintf('frame_%04d_sc%03d.mat', ...
                        global_t_pos, subcarrier_idx));
                    save(save_name, 'Distorted_CDM', 'Label_Bits', ...
                        'modulation', 'Frame_ID', 'Split', ...
                        'Subcarrier_ID', 'FFT_Bin', 'Symbols_Per_Sample', ...
                        'Source_Subfolder', 'Source_File');

                    total_saved_count = total_saved_count + 1;
                    if strcmp(Split, 'train')
                        train_saved_count = train_saved_count + 1;
                    else
                        val_saved_count = val_saved_count + 1;
                    end
                end

                fprintf('  [Saved:%s] frame %03d -> %d independent CDM samples\n', ...
                    Split, global_t_pos, total_active_subcarriers);
            catch ME
                fprintf('  [Skipped] %s: %s\n', bin_file, ME.message);
                total_skipped_count = total_skipped_count + 1;
            end
        end
    end
end

expected_train = numel(mod_list) * numel(train_frame_ids) * ...
    total_active_subcarriers;
expected_val = numel(mod_list) * numel(val_frame_ids) * ...
    total_active_subcarriers;

fprintf('\n============================================================\n');
fprintf('Independent-subcarrier Distorted CDM generation completed.\n');
fprintf('Saved feature files: %d (expected %d)\n', ...
    total_saved_count, expected_train + expected_val);
fprintf('Training samples: %d (expected %d)\n', ...
    train_saved_count, expected_train);
fprintf('Validation samples: %d (expected %d)\n', ...
    val_saved_count, expected_val);
fprintf('Skipped received signals: %d\n', total_skipped_count);
fprintf('Dataset directory: %s\n', dataset_root);
fprintf('Sample shape: 1 x %d x %d, one subcarrier per file.\n', ...
    cdm_size, cdm_size);
fprintf('============================================================\n');

function signal = readKeysightBin(file_path)
    fid = fopen(file_path, 'rb');
    if fid == -1
        error('Cannot open received signal file.');
    end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fread(fid, 2, '*char');
    fread(fid, 2, '*char');
    fread(fid, 1, 'int32');
    fread(fid, 1, 'int32');
    fread(fid, 1, 'int32');
    fread(fid, 1, 'int32');
    fread(fid, 1, 'int32');
    num_points = fread(fid, 1, 'int32');
    fread(fid, 1, 'int32');
    fread(fid, 1, 'float32');
    fread(fid, 1, 'float64');
    fread(fid, 1, 'float64');
    fread(fid, 1, 'float64');
    fread(fid, 1, 'int32');
    fread(fid, 1, 'int32');
    fread(fid, 16, '*char');
    fread(fid, 16, '*char');
    fread(fid, 24, '*char');
    fread(fid, 16, '*char');
    fread(fid, 1, 'float64');
    fread(fid, 1, 'uint32');
    fread(fid, 1, 'int32');
    fread(fid, 1, 'int16');
    bytes_per_point = fread(fid, 1, 'int16');
    fread(fid, 1, 'int32');

    if isempty(num_points) || num_points <= 0
        error('Invalid point count in binary header.');
    end

    switch bytes_per_point
        case 4
            signal = fread(fid, num_points, 'float32').';
        case 2
            signal = fread(fid, num_points, 'int16').';
        case 1
            signal = fread(fid, num_points, 'int8').';
        otherwise
            signal = fread(fid, num_points, 'double').';
    end

    if numel(signal) ~= num_points
        error('Binary signal is incomplete.');
    end
end

function cdm = generate_CDM_Smooth(complex_symbols, grid_size)
    edges = linspace(-2.0, 2.0, grid_size + 1);
    counts = histcounts2(real(complex_symbols), imag(complex_symbols), ...
        edges, edges);
    cdm = rot90(counts);
    cdm = imgaussfilt(cdm, 1.0);
    peak = max(cdm(:));
    if peak > 0
        cdm = cdm / peak;
    end
    cdm = single(cdm);
end
