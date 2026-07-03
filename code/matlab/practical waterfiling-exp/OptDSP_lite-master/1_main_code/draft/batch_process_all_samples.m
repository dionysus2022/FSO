%% batch_process_all_samples.m
% Batch SNR Pipeline: read all samples, compute 3-level SNR, plot histogram
% Sync: signal_pipeline_v1 style (LTS peak, with zeros_head/n_guard backup)

clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

global PROG;
PROG.showMessagesLevel = 0;
initProg();

%% ===================== Config =====================

cfg = struct();
cfg.data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
cfg.rx_date   = '2026.06.26';
cfg.tx_root   = fullfile(cfg.data_root, 'tx_3frame_6mod');
cfg.out_root  = fullfile(cfg.data_root, 'dataset_batch_snr');

cfg.Fs_rx   = 80e9;
cfg.Fs_base = 16e9;
cfg.n_frames = 3;
cfg.M_time = 32768;
cfg.cdm_bins = 64;
cfg.cdm_clip = 3.0;

cfg.mod_names = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};
cfg.sub_list  = {'sub1','sub2','sub3'};

cfg.turb_map = containers.Map('KeyType','char','ValueType','char');
cfg.turb_map('sub1') = 'weak';
cfg.turb_map('sub2') = 'moderate';
cfg.turb_map('sub3') = 'strong';

% OFDM params
cfg.zeros_head = 80;
cfg.n_fft = 256;
cfg.n_guard = 16;
cfg.n_syms = 128;
cfg.carrier_loc = 4:126;
cfg.n_sc = length(cfg.carrier_loc);
cfg.sym_len = cfg.n_fft + cfg.n_guard;
cfg.header_len_16 = cfg.zeros_head + cfg.n_guard + 2*cfg.n_fft;
cfg.frame_len_16 = cfg.header_len_16 + cfg.sym_len * cfg.n_syms;

% Sync params
cfg.sync_decim = 20;
cfg.num_corr_candidates = 60;
cfg.frame_margin_80 = 8000;
cfg.fine_search_len_80 = 30000;

LTS = make_lts_local(cfg.n_fft);

%% ===================== Statistics =====================

all_frame_snr = [];
all_file_snr = [];
all_sc_snr = [];
all_mod_labels = {};
all_turb_labels = {};

snr_file_results = table();

stats = struct();
stats.total_files   = 0;
stats.total_frames  = 0;
stats.ok_read       = 0;
stats.ok_sync       = 0;
stats.ok_demod      = 0;
stats.fail_read     = 0;
stats.fail_sync     = 0;
stats.fail_demod    = 0;

%% ===================== Main Loop =====================

for mi = 1:length(cfg.mod_names)
    mod_name = cfg.mod_names{mi};
    [Mq, bits] = mod_to_order_bits_local(mod_name);
    nBpS_net = bits - 0.2 * (bits > 2);

    TX.SIG = setSignalParams('symRate', 8e9, 'M', Mq, ...
        'nPol', 1, 'nBpS', nBpS_net, 'nSyms', cfg.n_syms, ...
        'roll-off', 0.25, 'modulation', 'QAM');
    TX.QAM = QAM_config(TX.SIG);
    C = TX.QAM.IQmap;

    rx_mod_dir = fullfile(cfg.data_root, 'rx_data', cfg.rx_date, mod_name);
    if ~exist(rx_mod_dir, 'dir'), continue; end

    for si = 1:length(cfg.sub_list)
        sub_name = cfg.sub_list{si};
        if isKey(cfg.turb_map, sub_name)
            turb_name = cfg.turb_map(sub_name);
        else
            turb_name = sub_name;
        end

        rx_dir = fullfile(rx_mod_dir, sub_name);
        if ~exist(rx_dir, 'dir'), continue; end
        bin_list = dir(fullfile(rx_dir, '*.bin'));
        if isempty(bin_list), continue; end

        for bi = 1:length(bin_list)
            [~, fname] = fileparts(bin_list(bi).name);
            sig_idx = str2double(fname);
            if isnan(sig_idx) || sig_idx < 1, continue; end

            rx_bin  = fullfile(rx_dir, bin_list(bi).name);
            tx_txt  = fullfile(cfg.tx_root, mod_name, sub_name, sprintf('sig_%04d.txt', sig_idx));

            stats.total_files = stats.total_files + 1;

            % ---- Read RX .bin ----
            try
                rx80 = read_keysight_bin_local(rx_bin);
                rx80 = rx80(:).';
                rx80 = rx80 - mean(rx80);
                rx80 = rx80 ./ (rms(rx80) + eps);
                stats.ok_read = stats.ok_read + 1;
            catch
                stats.fail_read = stats.fail_read + 1;
                continue;
            end

            % ---- TX txt ----
            if ~exist(tx_txt, 'file'), continue; end
            try
                tx_ref80 = load_ascii_complex_local(tx_txt);
                tx_ref80 = tx_ref80(:);
                frame_len_80 = floor(length(tx_ref80) / cfg.n_frames);
                if frame_len_80 <= 0, continue; end
            catch
                continue;
            end

            % ---- TX refs ----
            tx_refs = cell(1, cfg.n_frames);
            ref_ok = true;
            for tid = 1:cfg.n_frames
                ref_file = fullfile(cfg.tx_root, mod_name, sub_name, ...
                    sprintf('sig_%04d_frame%d.mat', sig_idx, tid));
                if ~exist(ref_file, 'file'), ref_ok = false; break; end
                tmp_ref = load(ref_file);
                if ~isfield(tmp_ref, 'data_tx'), ref_ok = false; break; end
                tx_refs{tid} = tmp_ref.data_tx.';
            end
            if ~ref_ok, continue; end

            % ---- Coarse sync ----
            try
                [rx_start_80, start_tx_id, ~] = find_awg_cycle_start_local( ...
                    rx80, tx_ref80, frame_len_80, cfg);
                stats.ok_sync = stats.ok_sync + 1;
            catch
                stats.fail_sync = stats.fail_sync + 1;
                continue;
            end

            % ---- Frame processing ----
            file_frame_snr = [];
            file_sc_snr = [];

            for rk = 1:cfg.n_frames
                seg_start_nom = rx_start_80 + (rk-1) * frame_len_80;
                try
                    [rx_raw80, ~, ~] = extract_frame_with_margin_local( ...
                        rx80, seg_start_nom, frame_len_80, cfg.frame_margin_80);

                    rx_raw16 = resample(rx_raw80(:).', cfg.Fs_base, cfg.Fs_rx);
                    rx_raw16 = rx_raw16 - mean(rx_raw16);
                    rx_raw16 = rx_raw16 ./ (rms(rx_raw16) + eps);

                    % LTS sync: signal_pipeline_v1 style
                    LTS_t = LTS.time(:).';
                    xc = abs(conv(rx_raw16, flipud(conj(LTS_t)), 'valid'));
                    [~, pk] = max(xc);
                    frm_start = pk;

                    frame_len_16_expected = max(1, round(frame_len_80 * cfg.Fs_base / cfg.Fs_rx));

                    if frm_start + frame_len_16_expected > length(rx_raw16)
                        error('frame too short');
                    end

                    rx_frame16 = rx_raw16(frm_start : frm_start + frame_len_16_expected - 1);

                    [rx_sc, ~] = demod_one_frame_local_simple(rx_frame16, LTS, cfg);

                    % 3-level SNR
                    sc_snr = 10 * log10(mean(abs(rx_sc).^2, 2));
                    frame_snr = 10 * log10(mean(abs(rx_sc(:)).^2));

                    file_frame_snr(end+1) = frame_snr;
                    file_sc_snr = [file_sc_snr, sc_snr(:)];

                    all_frame_snr(end+1) = frame_snr;
                    all_sc_snr = [all_sc_snr, sc_snr(:)];
                    all_mod_labels{end+1} = mod_name;
                    all_turb_labels{end+1} = turb_name;

                    stats.ok_demod = stats.ok_demod + 1;
                catch
                    stats.fail_demod = stats.fail_demod + 1;
                end
            end

            % File-level SNR
            if ~isempty(file_frame_snr)
                file_snr = 10 * log10(mean(10.^(file_frame_snr/10)));
                all_file_snr(end+1) = file_snr;

                snr_file_results = [snr_file_results; table(...
                    {mod_name}, {sub_name}, {turb_name}, sig_idx, ...
                    file_snr, mean(file_frame_snr), length(file_frame_snr), ...
                    'VariableNames', {'Mod', 'Sub', 'Turb', 'SigIdx', ...
                    'FileSNR_dB', 'MeanFrameSNR_dB', 'NFrames'})];
            end

            stats.total_frames = stats.total_frames + length(file_frame_snr);

            if mod(bi, 15) == 0 || bi == length(bin_list)
                fprintf('%s/%s: %d/%d OK, frames=%d\n', ...
                    mod_name, sub_name, bi, length(bin_list), stats.ok_demod);
            end
        end
    end
end

%% ===================== Results =====================

fprintf('\n============================================\n');
fprintf('  Batch SNR Processing Complete\n');
fprintf('============================================\n');
fprintf('Files: %d OK, %d fail_read, %d fail_sync, %d fail_demod\n', ...
    stats.ok_read, stats.fail_read, stats.fail_sync, stats.fail_demod);
fprintf('Frames demod OK: %d\n', stats.ok_demod);
fprintf('--------------------------------------------\n');

if isempty(all_frame_snr)
    fprintf('No successful frames.\n');
    return;
end

fprintf('\n--- Level 1: Subcarrier SNR ---\n');
mean_sc = mean(all_sc_snr, 2);
fprintf('  123 subcarrier SNR range: %.2f ~ %.2f dB\n', min(mean_sc), max(mean_sc));

fprintf('\n--- Level 2: Frame SNR ---\n');
fprintf('  Mean: %.2f dB, Median: %.2f dB, Std: %.2f dB\n', ...
    mean(all_frame_snr), median(all_frame_snr), std(all_frame_snr));
fprintf('  Min: %.2f dB, Max: %.2f dB\n', min(all_frame_snr), max(all_frame_snr));

fprintf('\n--- Level 3: File SNR ---\n');
fprintf('  Mean: %.2f dB, Median: %.2f dB, Std: %.2f dB\n', ...
    mean(all_file_snr), median(all_file_snr), std(all_file_snr));

%% ===================== Save =====================

snr_summary = struct();
snr_summary.frame_snr = all_frame_snr;
snr_summary.file_snr = all_file_snr;
snr_summary.sc_snr = all_sc_snr;
snr_summary.frame_mean = mean(all_frame_snr);
snr_summary.frame_median = median(all_frame_snr);
snr_summary.frame_std = std(all_frame_snr);
snr_summary.file_mean = mean(all_file_snr);
snr_summary.file_median = median(all_file_snr);
snr_summary.file_std = std(all_file_snr);
snr_summary.mod_labels = all_mod_labels;
snr_summary.turb_labels = all_turb_labels;

out_dir = fullfile(cfg.out_root, 'snr_results');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

save(fullfile(out_dir, 'snr_summary.mat'), 'snr_summary');
writetable(snr_file_results, fullfile(out_dir, 'snr_per_file.csv'));
fprintf('\nResults saved to: %s\n', out_dir);

%% ===================== Histogram =====================

figure('Position', [100 100 1400 900]);

subplot(2, 3, 1);
histogram(all_frame_snr, 50, 'FaceColor', '#0072BD', 'EdgeColor', 'none');
xlabel('Frame SNR (dB)');
ylabel('Count');
title(sprintf('Frame SNR Distribution (n=%d, mean=%.2f dB)', ...
    length(all_frame_snr), mean(all_frame_snr)));
grid on;

subplot(2, 3, 2);
hold on;
mods = unique(all_mod_labels);
colors = lines(length(mods));
for i = 1:length(mods)
    mask = strcmp(all_mod_labels, mods{i});
    histogram(all_frame_snr(mask), 30, 'FaceColor', colors(i,:), ...
        'EdgeColor', 'none', 'FaceAlpha', 0.5);
end
legend(mods, 'Location', 'northwest');
xlabel('Frame SNR (dB)');
ylabel('Count');
title('Frame SNR by Modulation');
grid on;

subplot(2, 3, 3);
histogram(all_file_snr, 40, 'FaceColor', '#D95319', 'EdgeColor', 'none');
xlabel('File SNR (dB)');
ylabel('Count');
title(sprintf('File SNR Distribution (n=%d, mean=%.2f dB)', ...
    length(all_file_snr), mean(all_file_snr)));
grid on;

subplot(2, 3, 4);
imagesc(all_sc_snr);
colorbar;
xlabel('Frame Index');
ylabel('Subcarrier Index');
title(sprintf('Per-Subcarrier SNR (%d frames)', size(all_sc_snr, 2)));
colormap('jet');
set(gca, 'YDir', 'normal');

subplot(2, 3, 5);
[f, x] = ecdf(all_frame_snr);
plot(x, f*100, 'LineWidth', 2, 'Color', '#0072BD');
xlabel('Frame SNR (dB)');
ylabel('Cumulative (%)');
title('Cumulative SNR Distribution');
grid on;
ylim([0 100]);
hold on;
for thresh = [0 5 10 15 20]
    pct = sum(all_frame_snr >= thresh) / length(all_frame_snr) * 100;
    plot(thresh, 100-pct, 'ro', 'MarkerSize', 6);
    text(thresh+0.3, 100-pct+2, sprintf('%.1f%% > %ddB', 100-pct, thresh), ...
        'FontSize', 8, 'Color', 'r');
end

subplot(2, 3, 6);
boxchart(categorical(all_mod_labels), all_frame_snr);
xlabel('Modulation');
ylabel('Frame SNR (dB)');
title('Frame SNR Boxplot by Modulation');
grid on;

saveas(gcf, fullfile(out_dir, 'snr_histogram.png'));
fprintf('Histogram saved to: %s\n', fullfile(out_dir, 'snr_histogram.png'));

fprintf('\n===== Batch SNR Analysis Complete =====\n');

%% =====================================================================
%%                   Helper Functions
%% =====================================================================

function [Mq, bits] = mod_to_order_bits_local(mod_name)
    switch mod_name
        case 'QPSK',    Mq = 4;   bits = 2;
        case '16QAM',   Mq = 16;  bits = 4;
        case '32QAM',   Mq = 32;  bits = 5;
        case '64QAM',   Mq = 64;  bits = 6;
        case '128QAM',  Mq = 128; bits = 7;
        case '256QAM',  Mq = 256; bits = 8;
        otherwise, error('unknown mod: %s', mod_name);
    end
end

%% Keysight .bin reader (complex I/Q)
function y = read_keysight_bin_local(filename)
    fid = fopen(filename, 'rb');
    if fid == -1, error('Cannot open: %s', filename); end
    fread(fid, 2, '*char')'; fread(fid, 2, '*char')';
    fread(fid, 1, 'int32'); fread(fid, 1, 'int32');
    fread(fid, 1, 'int32'); fread(fid, 1, 'int32');
    fread(fid, 1, 'int32'); num_points = fread(fid, 1, 'int32');
    fread(fid, 1, 'int32'); fread(fid, 1, 'float32');
    fread(fid, 1, 'float64'); fread(fid, 1, 'float64');
    fread(fid, 1, 'float64'); fread(fid, 1, 'int32');
    fread(fid, 1, 'int32'); fread(fid, 16, '*char')';
    fread(fid, 16, '*char')'; fread(fid, 24, '*char')';
    fread(fid, 16, '*char')'; fread(fid, 1, 'float64');
    fread(fid, 1, 'uint32'); fread(fid, 1, 'int32');
    fread(fid, 1, 'int16'); bpp = fread(fid, 1, 'int16');
    fread(fid, 1, 'int32');
    switch bpp
        case 4, raw = fread(fid, num_points, 'float32');
        case 2, raw = fread(fid, num_points, 'int16');
        case 1, raw = fread(fid, num_points, 'int8');
        otherwise, raw = fread(fid, num_points, 'double');
    end
    fclose(fid);
    raw = raw(:);
    n = floor(length(raw)/2);
    raw = raw(1:2*n);
    y = double(raw(1:2:end)) + 1j * double(raw(2:2:end));
end

function x = load_ascii_complex_local(filename)
    tmp = load(filename);
    if size(tmp, 2) >= 2
        x = complex(tmp(:,1), tmp(:,2));
    else
        x = tmp(:);
    end
end

function LTS = make_lts_local(n_fft)
    load('LongTrainSym_ini.mat', 'LongTrainSym_ini');
    LTS_f = LongTrainSym_ini(1:n_fft);
    LTS_f([1 n_fft/2+1]) = 0;
    ltrs_in = LTS_f;
    ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));
    LTS.freq = LTS_f(:);
    LTS.time = ifft(ltrs_in(:));
end

function [start_80, start_tx_id, metric_best] = find_awg_cycle_start_local(rx80, tx_ref80, frame_len_80, cfg)
    rx80 = rx80(:); tx_ref80 = tx_ref80(:);
    if length(tx_ref80) < cfg.n_frames * frame_len_80, error('tx_ref80 too short'); end
    if length(rx80) < cfg.n_frames * frame_len_80, error('rx80 too short'); end
    decim = cfg.sync_decim;
    rx_env = abs(rx80(1:decim:end)); rx_env = rx_env - mean(rx_env); rx_env = rx_env ./ (std(rx_env) + eps);
    all_score = []; all_idx = []; all_tid = [];
    for tid = 1:cfg.n_frames
        tx_frame = tx_ref80((tid-1)*frame_len_80 + (1:frame_len_80));
        tx_env = abs(tx_frame(1:decim:end));
        max_len = min(length(tx_env), 12000);
        tx_env = tx_env - mean(tx_env); tx_env = tx_env ./ (std(tx_env) + eps);
        if length(rx_env) < length(tx_env), continue; end
        c = conv(rx_env, flipud(tx_env), 'valid'); c_abs = abs(c);
        n_take = min(cfg.num_corr_candidates, length(c_abs));
        [vals, idxs] = maxk(c_abs, n_take);
        all_score = [all_score; vals(:)]; all_idx = [all_idx; idxs(:)];
        all_tid = [all_tid; tid * ones(n_take,1)];
    end
    if isempty(all_score), error('no correlation candidate'); end
    [~, order] = sort(all_score, 'descend');
    for ii = 1:length(order)
        idx_d = all_idx(order(ii)); tid = all_tid(order(ii));
        coarse_start = (idx_d - 1)*decim + 1;
        start_candidate = refine_start_fullrate_env_local(rx80, tx_ref80, frame_len_80, tid, coarse_start, cfg);
        tx_id_candidate = tid;
        while start_candidate + cfg.n_frames*frame_len_80 - 1 > length(rx80)
            start_candidate = start_candidate - frame_len_80;
            tx_id_candidate = mod(tx_id_candidate - 2, cfg.n_frames) + 1;
        end
        while start_candidate < 1
            start_candidate = start_candidate + frame_len_80;
            tx_id_candidate = mod(tx_id_candidate, cfg.n_frames) + 1;
        end
        if start_candidate >= 1 && start_candidate + cfg.n_frames*frame_len_80 - 1 <= length(rx80)
            start_80 = round(start_candidate); start_tx_id = tx_id_candidate;
            metric_best = all_score(order(ii)); return;
        end
    end
    error('cannot contain 3 complete frames');
end

function start_refined = refine_start_fullrate_env_local(rx80, tx_ref80, frame_len_80, tx_id, coarse_start, cfg)
    tpl = tx_ref80((tx_id-1)*frame_len_80 + (1:frame_len_80));
    L = min([length(tpl), cfg.fine_search_len_80, length(rx80)]);
    tpl_env = abs(tpl(1:L)); tpl_env = tpl_env - mean(tpl_env); tpl_env = tpl_env ./ (std(tpl_env) + eps);
    win = 4 * cfg.sync_decim;
    s1 = max(1, coarse_start - win); s2 = min(length(rx80) - L + 1, coarse_start + win);
    if s2 < s1, start_refined = coarse_start; return; end
    best_val = -inf; best_s = coarse_start;
    for s = s1:s2
        r_env = abs(rx80(s:s+L-1)); r_env = r_env - mean(r_env); r_env = r_env ./ (std(r_env) + eps);
        val = abs(r_env(:)' * tpl_env(:));
        if val > best_val, best_val = val; best_s = s; end
    end
    start_refined = best_s;
end

function [rx_raw80, seg_start, seg_end] = extract_frame_with_margin_local(rx80, seg_start_nom, frame_len_80, margin)
    seg_start = max(1, seg_start_nom - margin);
    seg_end = min(length(rx80), seg_start_nom + frame_len_80 - 1 + margin);
    if seg_end <= seg_start, error('invalid segment range'); end
    rx_raw80 = rx80(seg_start:seg_end);
end

%% Demodulation (signal_pipeline_v1 style)
function [rx_sc, info] = demod_one_frame_local_simple(rx_frame16, LTS, cfg)
    rx = rx_frame16(:).';
    n_fft = cfg.n_fft; n_guard = cfg.n_guard; sym_len = cfg.sym_len;

    LTS_t = LTS.time(:).';
    xc = abs(conv(rx, flipud(conj(LTS_t)), 'valid'));
    [~, lts_peak] = max(xc);

    frm_start = lts_peak;
    remaining = length(rx) - frm_start + 1;

    if remaining < 2 * n_fft
        error('frame too short: remaining=%d', remaining);
    end

    lts1 = rx(frm_start : frm_start + n_fft - 1);
    lts2 = rx(frm_start + n_fft : frm_start + 2*n_fft - 1);

    cfo = angle(sum(lts1(:).*conj(lts2(:))))/(2*pi*n_fft);
    rx_comp = rx(frm_start:end) .* exp(-1j*2*pi*cfo*(0:remaining-1)/n_fft);

    data_start = 2 * n_fft + 1;
    dp = rx_comp(data_start:end);
    nd = floor(length(dp) / sym_len);

    if nd < 1, error('no complete symbols'); end

    dp = dp(1:nd*sym_len);
    dm = reshape(dp, sym_len, nd);
    dn = dm(n_guard+1:end, :);
    fd = fft(dn, n_fft, 1) / sqrt(n_fft);

    lts_avg = (lts1(:) + lts2(:)) / 2;
    lts_fd = fft(lts_avg, n_fft) / sqrt(n_fft);
    H = lts_fd ./ (LTS.freq(:) + 1e-12);
    H(abs(LTS.freq(:)) < 0.5) = 1;
    feq = fd ./ H;

    rx_sc = feq(cfg.carrier_loc, :);
    info = struct();
    info.cfo = cfo;
    info.n_use = nd;
end
