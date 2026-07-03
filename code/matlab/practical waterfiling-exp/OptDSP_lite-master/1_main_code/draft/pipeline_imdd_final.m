%% pipeline_imdd_final.m
% Final pipeline for IM/DD APD direct-detection experiment
% Sync: iterative_rx1_style | SNR: rx1-style (symDemapper + EVM_eval)

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
cfg.out_root  = fullfile(cfg.data_root, 'dataset_imdd_final');

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

cfg.SIG.nSyms = 128;
cfg.ofdm.NumberOfIFFTSamples = 256;
cfg.ofdm.NumberOfGuardTime   = 16;
cfg.ofdm.Carrier_location    = 4:126;
cfg.ofdm.Carrier_location_demo = [4:126, 132:254];
cfg.ofdm.NumberOfCarriers = length(cfg.ofdm.Carrier_location);
cfg.ofdm.NumberOfCarriers_demo = length(cfg.ofdm.Carrier_location_demo);
cfg.ofdm.size = cfg.SIG.nSyms;

cfg.zeros_head = 80;
cfg.n_fft   = cfg.ofdm.NumberOfIFFTSamples;
cfg.n_guard = cfg.ofdm.NumberOfGuardTime;
cfg.sym_len = cfg.n_fft + cfg.n_guard;
cfg.frame_len_16 = cfg.zeros_head + cfg.n_guard + 2*cfg.n_fft + cfg.sym_len*cfg.SIG.nSyms;
cfg.header_len_16 = cfg.zeros_head + cfg.n_guard + 2*cfg.n_fft;
cfg.carrier_loc = 4:126;
cfg.n_sc = length(cfg.carrier_loc);

cfg.sync_decim = 20;
cfg.num_corr_candidates = 60;
cfg.frame_margin_80 = 8000;
cfg.fine_search_len_80 = 30000;

%% ===================== Output directories =====================

out_time = fullfile(cfg.out_root, 'time_32768');
out_freq = fullfile(cfg.out_root, 'freq_sc');
out_cdm  = fullfile(cfg.out_root, 'cdm_64');
make_dir(out_time); make_dir(out_freq); make_dir(out_cdm);

%% ===================== Statistics =====================

index_rows = {};
all_frame_snr = [];
all_frame_power = [];
all_file_snr = [];
all_file_power = [];
all_mod_labels = {};
all_turb_labels = {};

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
    label_id = mi - 1;
    [Mq, bits] = mod_to_order_bits_local(mod_name);
    nBpS_net = bits - 0.2 * (bits > 2);

    TX.SIG = setSignalParams('symRate', 8e9, 'M', Mq, ...
        'nPol', 1, 'nBpS', nBpS_net, 'nSyms', cfg.SIG.nSyms, ...
        'roll-off', 0.25, 'modulation', 'QAM');
    TX.QAM = QAM_config(TX.SIG);
    C = TX.QAM.IQmap;
    DSP = struct();
    DSP.DEMAPPER.normMethod = 'MMSE';

    rx_mod_dir = fullfile(cfg.data_root, 'rx_data', cfg.rx_date, mod_name);
    if ~exist(rx_mod_dir, 'dir'), continue; end

    for si = 1:length(cfg.sub_list)
        sub_name = cfg.sub_list{si};
        if isKey(cfg.turb_map, sub_name)
            turb_name = cfg.turb_map(sub_name);
        else
            turb_name = sub_name;
        end

        make_dir(fullfile(out_time, mod_name, turb_name));
        make_dir(fullfile(out_freq, mod_name, turb_name));
        make_dir(fullfile(out_cdm,  mod_name, turb_name));

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

            % ---- Read RX .bin (real-valued, IM/DD) ----
            try
                [rx80, ~] = read_keysight_bin_robust_real_local(rx_bin);
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

            % ---- Resample 80G -> 16G ----
            rx16 = resample(rx80, cfg.Fs_base, cfg.Fs_rx);
            rx16 = rx16 - mean(rx16);
            rx16 = rx16 ./ (mean(abs(rx16)) + eps);
            wrap_len = min(length(rx16), 3*cfg.frame_len_16);
            rx16_ext = [rx16, rx16(1:wrap_len)];

            % ---- Recursive frame extraction ----
            cursor = 1;
            file_frame_snr = [];
            file_frame_power = [];

            for rk = 1:cfg.n_frames
                search_sig = rx16_ext(cursor:end);
                if length(search_sig) < cfg.frame_len_16, break; end

                try
                    [lts_start_rel, frame_start_rel, ~] = find_one_frame_start_rx1_style( ...
                        search_sig, cfg.ofdm, cfg.SIG.nSyms, cfg.zeros_head);
                catch
                    cursor = cursor + round(cfg.frame_len_16 * 0.5);
                    stats.fail_sync = stats.fail_sync + 1;
                    continue;
                end

                lts_start_abs   = cursor + lts_start_rel - 1;
                frame_start_abs = cursor + frame_start_rel - 1;

                % Demod
                try
                    [rx_sc, demod_info] = demod_one_frame_from_lts_start( ...
                        rx16_ext, lts_start_abs, cfg.ofdm, cfg.SIG.nSyms);
                catch
                    cursor = frame_start_abs + cfg.frame_len_16;
                    stats.fail_demod = stats.fail_demod + 1;
                    continue;
                end

                % rx1-style SNR (match against 3 TX refs, pick best)
                [best_tx_id, snr_frame_rx1, snr_sc_rx1, snr_ref_list, txafdem, ber_sc] = ...
                    match_rx_to_tx_by_rx1_snr_local(rx_sc, tx_refs, C, DSP);

                % Power SNR (diagnostic)
                power_snr = 10 * log10(mean(abs(rx_sc(:)).^2));

                % Save features
                frame_end_abs = frame_start_abs + cfg.frame_len_16 - 1;
                if frame_end_abs <= length(rx16_ext)
                    rx_frame = rx16_ext(frame_start_abs : frame_end_abs);
                else
                    rx_frame = rx16_ext(frame_start_abs : end);
                end

                file_id = sprintf('%s_%s_sig%04d', mod_name, sub_name, sig_idx);
                base_name = sprintf('sig_%04d_rxframe%d_tx%d', sig_idx, rk, best_tx_id);

                rx_time = make_time_payload_local(rx_frame, cfg);
                cdm64 = make_cdm_local(rx_sc, cfg.cdm_bins, cfg.cdm_clip);

                out_time_file = fullfile(out_time, mod_name, turb_name, [base_name '.mat']);
                out_freq_file = fullfile(out_freq, mod_name, turb_name, [base_name '.mat']);
                out_cdm_file  = fullfile(out_cdm,  mod_name, turb_name, [base_name '.mat']);

                sample_time = struct();
                sample_time.rx_time = single(rx_time(:));
                sample_time.label_id = label_id;
                sample_time.label_name = mod_name;
                sample_time.sig_idx = sig_idx;
                sample_time.rx_frame_idx = rk;
                sample_time.best_tx_frame_id = best_tx_id;
                sample_time.sub_name = sub_name;
                sample_time.turbulence = turb_name;
                save(out_time_file, 'sample_time', '-v7.3');

                sample_freq = struct();
                sample_freq.rx_sc = single(rx_sc);
                sample_freq.label_id = label_id;
                sample_freq.label_name = mod_name;
                sample_freq.sig_idx = sig_idx;
                sample_freq.rx_frame_idx = rk;
                sample_freq.best_tx_frame_id = best_tx_id;
                sample_freq.sub_name = sub_name;
                sample_freq.turbulence = turb_name;
                sample_freq.snr_frame_rx1_db = snr_frame_rx1;
                sample_freq.snr_sc_rx1_db = single(snr_sc_rx1);
                sample_freq.snr_ref_list_db = snr_ref_list;
                sample_freq.txafdem_matrix = single(txafdem);
                sample_freq.ber_sc = single(ber_sc);
                save(out_freq_file, 'sample_freq', '-v7.3');

                sample_cdm = struct();
                sample_cdm.cdm64 = single(cdm64);
                sample_cdm.label_id = label_id;
                sample_cdm.label_name = mod_name;
                sample_cdm.sig_idx = sig_idx;
                sample_cdm.rx_frame_idx = rk;
                sample_cdm.best_tx_frame_id = best_tx_id;
                sample_cdm.sub_name = sub_name;
                sample_cdm.turbulence = turb_name;
                save(out_cdm_file, 'sample_cdm', '-v7.3');

                % Index row
                sc_snr_power = 10 * log10(mean(abs(rx_sc).^2, 2));
                ber_valid = ber_sc(isfinite(ber_sc));
                index_rows{end+1} = struct( ...
                    'out_time', out_time_file, ...
                    'out_freq', out_freq_file, ...
                    'out_cdm',  out_cdm_file, ...
                    'file_id', file_id, ...
                    'sig_idx', sig_idx, ...
                    'frame_idx', rk, ...
                    'best_tx_frame_id', best_tx_id, ...
                    'label_id', label_id, ...
                    'label_name', mod_name, ...
                    'mod_order', Mq, ...
                    'sub_name', sub_name, ...
                    'turbulence', turb_name, ...
                    'SNRFrameRX1_dB', snr_frame_rx1, ...
                    'SNRFramePower_dB', power_snr, ...
                    'SNRScMeanRX1_dB', nanmean(snr_sc_rx1(isfinite(snr_sc_rx1))), ...
                    'SNRScStdRX1_dB', nanstd(snr_sc_rx1(isfinite(snr_sc_rx1))), ...
                    'SNRSC_mean_dB', mean(sc_snr_power), ...
                    'SNRSC_std_dB', std(sc_snr_power), ...
                    'SNRSC_min_dB', min(sc_snr_power), ...
                    'SNRSC_max_dB', max(sc_snr_power), ...
                    'N_ofdm_symbols', demod_info.n_use, ...
                    'BestTxFrameID', best_tx_id, ...
                    'BERMean', nanmean(ber_valid), ...
                    'valid_flag', true, ...
                    'valid_reason', 'ok');

                file_frame_snr(end+1) = snr_frame_rx1;
                file_frame_power(end+1) = power_snr;
                all_frame_snr(end+1) = snr_frame_rx1;
                all_frame_power(end+1) = power_snr;
                all_mod_labels{end+1} = mod_name;
                all_turb_labels{end+1} = turb_name;

                stats.ok_demod = stats.ok_demod + 1;
                cursor = frame_start_abs + cfg.frame_len_16;
            end

            stats.ok_sync = stats.ok_sync + 1;

            if ~isempty(file_frame_snr)
                all_file_snr(end+1) = 10*log10(mean(10.^(file_frame_snr/10)));
                all_file_power(end+1)  = 10*log10(mean(10.^(file_frame_power/10)));
            end

            stats.total_frames = stats.total_frames + length(file_frame_snr);

            if mod(bi, 15) == 0 || bi == length(bin_list)
                fprintf('%s/%s: %d/%d OK, frames=%d\n', ...
                    mod_name, sub_name, bi, length(bin_list), stats.ok_demod);
            end
        end
    end
end

%% ===================== Save dataset_index.csv =====================

if ~isempty(index_rows)
    index_table = struct2table([index_rows{:}]);
    index_file = fullfile(cfg.out_root, 'dataset_index.csv');
    writetable(index_table, index_file);
    fprintf('\ndataset_index.csv: %d rows\n', height(index_table));
else
    fprintf('\nNo frames.\n');
    return;
end

%% ===================== Summary =====================

fprintf('\n============================================\n');
fprintf('  IM/DD Final Dataset Complete\n');
fprintf('============================================\n');
fprintf('Files: %d OK, %d fail_read\n', stats.ok_read, stats.fail_read);
fprintf('Frames: %d / %d\n', stats.ok_demod, stats.total_files * 3);

fprintf('\n--- rx1-style SNR (primary) ---\n');
fprintf('  Frame: mean=%.2f median=%.2f std=%.2f dB\n', ...
    mean(all_frame_snr), median(all_frame_snr), std(all_frame_snr));

fprintf('\n--- Power SNR (diagnostic) ---\n');
fprintf('  Frame: mean=%.2f median=%.2f std=%.2f dB\n', ...
    mean(all_frame_power), median(all_frame_power), std(all_frame_power));

summary = struct();
summary.rx1_frame = all_frame_snr;
summary.power_frame = all_frame_power;
summary.rx1_file = all_file_snr;
summary.power_file = all_file_power;
summary.mod_labels = all_mod_labels;
summary.turb_labels = all_turb_labels;
summary.stats = stats;
save(fullfile(cfg.out_root, 'dataset_summary.mat'), 'summary');

%% ===================== Histograms =====================

figure('Position', [100 100 1400 900]);
subplot(2,3,1);
histogram(all_frame_snr, 50, 'FaceColor','#0072BD','EdgeColor','none');
xlabel('RX1 SNR (dB)'); ylabel('Count');
title(sprintf('RX1 SNR\nmean=%.2f dB', mean(all_frame_snr))); grid on;

subplot(2,3,2);
histogram(all_frame_power, 50, 'FaceColor','#D95319','EdgeColor','none');
xlabel('Power SNR (dB)'); ylabel('Count');
title(sprintf('Power SNR\nmean=%.2f dB', mean(all_frame_power))); grid on;

subplot(2,3,3); hold on;
mods = unique(all_mod_labels); colors = lines(length(mods));
for i = 1:length(mods)
    mask = strcmp(all_mod_labels, mods{i});
    histogram(all_frame_snr(mask), 30, 'FaceColor',colors(i,:), 'EdgeColor','none', 'FaceAlpha',0.5);
end
legend(mods, 'Location','northwest'); xlabel('RX1 SNR (dB)'); title('SNR by Mod'); grid on;

subplot(2,3,4);
histogram(all_file_snr, 40, 'FaceColor','#0072BD','EdgeColor','none');
xlabel('File RX1 SNR (dB)'); ylabel('Count');
title(sprintf('File SNR\nmean=%.2f dB', mean(all_file_snr))); grid on;

subplot(2,3,5);
ecdf(all_frame_snr); xlabel('RX1 SNR (dB)'); ylabel('Cumulative (%)'); grid on; ylim([0 100]);
title('Cumulative SNR');
hold on;
for thresh = [0 5 10 15 20]
    pct = sum(all_frame_snr >= thresh)/length(all_frame_snr)*100;
    plot(thresh, 100-pct, 'ro'); text(thresh+0.3, 100-pct+2, sprintf('%.0f%% > %d dB', 100-pct, thresh), 'FontSize',8);
end

subplot(2,3,6);
boxchart(categorical(all_mod_labels), all_frame_snr);
xlabel('Modulation'); ylabel('RX1 SNR (dB)'); title('SNR Boxplot'); grid on;

saveas(gcf, fullfile(cfg.out_root, 'snr_histogram.png'));
fprintf('Histogram: %s\n', fullfile(cfg.out_root, 'snr_histogram.png'));
fprintf('\n===== Done =====\n');

%% =====================================================================
%%                   Helper Functions
%% =====================================================================

function make_dir(d)
    if ~exist(d, 'dir'), mkdir(d); end
end

function [Mq,bits] = mod_to_order_bits_local(mod_name)
    switch mod_name
        case 'QPSK',    Mq=4;  bits=2;
        case '16QAM',   Mq=16; bits=4;
        case '32QAM',   Mq=32; bits=5;
        case '64QAM',   Mq=64; bits=6;
        case '128QAM',  Mq=128; bits=7;
        case '256QAM',  Mq=256; bits=8;
        otherwise, error('unknown mod: %s', mod_name);
    end
end

%% ===== Read real-valued Keysight .bin =====
function [y, info] = read_keysight_bin_robust_real_local(filename)
    info = struct(); info.method='unknown';
    try
        [y, info] = read_keysight_real_standard_or_infer_local(filename, false);
        info.method='standard_bpp'; return;
    catch ME1
        info.standard_error=ME1.message;
    end
    try
        [y, info] = read_keysight_real_standard_or_infer_local(filename, true);
        info.method='inferred_bpp'; return;
    catch ME2
        info.infer_error=ME2.message;
    end
    error('read failed: standard=[%s], infer=[%s]', info.standard_error, info.infer_error);
end

function [y, info] = read_keysight_real_standard_or_infer_local(filename, force_infer)
    fid = fopen(filename, 'rb', 'ieee-le');
    if fid == -1, error('Cannot open: %s', filename); end
    cleaner = onCleanup(@() fclose(fid));
    fread(fid,2,'*char')'; fread(fid,2,'*char')';
    fread(fid,1,'int32'); fread(fid,1,'int32');
    fread(fid,1,'int32'); fread(fid,1,'int32');
    fread(fid,1,'int32');
    num_points = fread(fid,1,'int32');
    fread(fid,1,'int32'); fread(fid,1,'float32');
    fread(fid,1,'float64'); fread(fid,1,'float64');
    fread(fid,1,'float64'); fread(fid,1,'int32');
    fread(fid,1,'int32'); fread(fid,16,'*char')';
    fread(fid,16,'*char')'; fread(fid,24,'*char')';
    fread(fid,16,'*char')'; fread(fid,1,'float64');
    fread(fid,1,'uint32'); fread(fid,1,'int32');
    fread(fid,1,'int16');
    bpp_read = fread(fid,1,'int16');
    buffer_size = fread(fid,1,'int32');
    data_start = ftell(fid);
    d = dir(filename);
    remain_bytes = d.bytes - data_start;
    if isempty(num_points) || numel(num_points)~=1 || num_points<=0, error('invalid num_points'); end
    bpp_candidates = [];
    if ~force_infer
        if isempty(bpp_read) || numel(bpp_read)~=1 || ~ismember(double(bpp_read),[1 2 4 8])
            error('invalid bpp');
        end
        bpp_candidates = double(bpp_read);
    else
        if ~isempty(bpp_read) && numel(bpp_read)==1 && ismember(double(bpp_read),[1 2 4 8])
            bpp_candidates(end+1) = double(bpp_read);
        end
        if ~isempty(buffer_size) && numel(buffer_size)==1 && buffer_size>0
            bpp_candidates(end+1) = round(double(buffer_size)/double(num_points));
        end
        if remain_bytes>0
            bpp_candidates(end+1) = round(double(remain_bytes)/double(num_points));
        end
        bpp_candidates = unique([bpp_candidates 4 2 1 8], 'stable');
        bpp_candidates = bpp_candidates(ismember(bpp_candidates, [1 2 4 8]));
    end
    last_msg = '';
    for ii = 1:length(bpp_candidates)
        bpp = bpp_candidates(ii);
        try
            fseek(fid, data_start, 'bof');
            switch bpp
                case 4, raw = fread(fid, num_points, 'float32');
                case 2, raw = fread(fid, num_points, 'int16');
                case 1, raw = fread(fid, num_points, 'int8');
                case 8, raw = fread(fid, num_points, 'double');
                otherwise, error('unsupported bpp');
            end
            raw = double(raw(:)).';
            if length(raw)<1000, error('too short'); end
            if std(raw(1:min(5000,end)))==0, error('zero variance'); end
            y = raw;
            info = struct();
            info.method='real'; info.bpp=double(bpp_read);
            info.inferred_bpp=bpp; info.num_points=double(num_points);
            info.buffer_size=double(buffer_size); info.data_start=data_start;
            return;
        catch ME
            last_msg = ME.message;
        end
    end
    error('all bpp candidates failed: %s', last_msg);
end

function x = load_ascii_complex_local(filename)
    tmp=load(filename);
    if size(tmp,2)>=2, x=complex(tmp(:,1),tmp(:,2));
    else, x=tmp(:); end
end

%% ===== rx1-style sync =====
function [lts_start, frame_start, info] = find_one_frame_start_rx1_style(rx, ofdm, n_syms, zeros_head)
    rx = rx(:).';
    n_fft   = ofdm.NumberOfIFFTSamples;
    n_guard = ofdm.NumberOfGuardTime;
    sym_len = n_fft + n_guard;
    symbol_bits = zeros_head + n_guard + 2*n_fft + sym_len*n_syms;
    search_len = min(length(rx), 2*symbol_bits);
    if search_len < symbol_bits, error('input too short'); end
    search_sig = rx(1:search_len);
    [detected_packet, edge_index] = packet_edge_power_dect(search_sig, zeros_head);
    load('LongTrainSym_ini.mat', 'LongTrainSym_ini');
    LTS_f = LongTrainSym_ini(1:n_fft); LTS_f([1 n_fft/2+1]) = 0;
    ltrs_in = LTS_f; ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));
    [fine_time_est, data_df, max_peak_long] = rx_fine_time_sync_cross_corr( ...
        detected_packet, n_guard, ltrs_in, zeros_head, 0);
    lts_start = edge_index + fine_time_est - 1;
    frame_start = lts_start - (zeros_head + n_guard);
    if frame_start < 1, frame_start = 1; end
    if lts_start < 1 || lts_start + 2*n_fft - 1 > length(rx), error('LTS out of range'); end
    info = struct();
    info.edge_index = edge_index; info.fine_time_est = fine_time_est;
    info.lts_start = lts_start; info.frame_start = frame_start;
    info.data_df = data_df; info.max_peak_long = max_peak_long;
    info.symbol_bits = symbol_bits;
end

%% ===== Demod from LTS start =====
function [rx_sc, info] = demod_one_frame_from_lts_start(rx, lts_start, ofdm, n_syms)
    rx = rx(:).';
    n_fft   = ofdm.NumberOfIFFTSamples;
    n_guard = ofdm.NumberOfGuardTime;
    sym_len = n_fft + n_guard;
    carrier_loc = 4:126;
    load('LongTrainSym_ini.mat', 'LongTrainSym_ini');
    LTS_f = LongTrainSym_ini(1:n_fft); LTS_f([1 n_fft/2+1]) = 0;
    ltrs_in = LTS_f; ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));
    LTS_f_ref = ltrs_in;
    lts1_end = lts_start + n_fft - 1; lts2_end = lts_start + 2*n_fft - 1;
    if lts_start < 1 || lts2_end > length(rx), error('frame too short for LTS'); end
    lts1 = rx(lts_start:lts1_end); lts2 = rx(lts_start+n_fft:lts2_end);
    pd = angle(sum(lts1(:).*conj(lts2(:)))); cfo = pd/(2*pi*n_fft);
    n = 0:length(rx)-1; rx_cfo = rx .* exp(-1j*2*pi*cfo*n/n_fft);
    lts1 = rx_cfo(lts_start:lts1_end); lts2 = rx_cfo(lts_start+n_fft:lts2_end);
    data_start = lts_start + 2*n_fft; data_end = data_start + sym_len*n_syms - 1;
    if data_end > length(rx_cfo)
        remain = length(rx_cfo)-data_start+1; nd = floor(remain/sym_len);
        if nd <= 0, error('no complete symbols'); end
        n_use = min(nd, n_syms);
    else
        n_use = n_syms;
    end
    data_end = data_start + sym_len*n_use - 1; dp = rx_cfo(data_start:data_end);
    dm = reshape(dp, sym_len, n_use); dn = dm(n_guard+1:end, :);
    fd = fft(dn, n_fft, 1) ./ sqrt(n_fft);
    lts_avg = (lts1(:)+lts2(:))/2; lts_fd = fft(lts_avg, n_fft) ./ sqrt(n_fft);
    H = lts_fd ./ (LTS_f_ref(:)+1e-12); H(abs(LTS_f_ref(:))<0.5) = 1;
    feq = fd ./ H; rx_sc = feq(carrier_loc, :);
    info = struct(); info.cfo = cfo; info.n_use = n_use;
end

%% ===== rx1-style SNR (symDemapper + EVM_eval) =====
function [snr_frame_db, snr_sc_db, txafdem_matrix, ber_sc] = ...
        compute_rx1_style_snr_local(rx_sc, tx_ref, C, DSP_template)
    rx_sc = double(rx_sc);
    tx_ref = double(tx_ref);
    n_sc = min(size(rx_sc,1), size(tx_ref,1));
    n_sym = min(size(rx_sc,2), size(tx_ref,2));
    rx_use = rx_sc(1:n_sc, 1:n_sym);
    tx_use = tx_ref(1:n_sc, 1:n_sym);
    txafdem_matrix = NaN(n_sc, n_sym);
    ber_sc = NaN(n_sc, 1);
    for sc = 1:n_sc
        DSP = DSP_template;
        DSP.DEMAPPER.N0 = 0;
        try
            [DSP.DEMAPPER, txafdem] = symDemapper( ...
                rx_use(sc,:), tx_use(sc,:), C, DSP.DEMAPPER);
            txafdem_matrix(sc,:) = txafdem;
            try
                [BER, ~] = BER_eval(DSP.DEMAPPER.txBits, DSP.DEMAPPER.rxBits);
                ber_sc(sc) = BER;
            catch
                ber_sc(sc) = NaN;
            end
        catch
            txafdem_matrix(sc,:) = NaN;
            ber_sc(sc) = NaN;
        end
    end
    try
        [~, snr_sc_db] = EVM_eval(rx_use, txafdem_matrix);
    catch
        snr_sc_db = NaN(n_sc, 1);
    end
    snr_sc_db = snr_sc_db(:);
    valid = snr_sc_db(isfinite(snr_sc_db) & snr_sc_db > -100);
    if isempty(valid)
        snr_frame_db = NaN;
    else
        snr_frame_db = 10 * log10(mean(10.^(valid/10)));
    end
end

%% ===== Match tx_frame1/2/3 by rx1-style SNR =====
function [best_tx_id, best_snr_db, best_snr_sc_db, ...
        snr_list_db, best_txafdem, best_ber_sc] = ...
        match_rx_to_tx_by_rx1_snr_local(rx_sc, tx_refs, C, DSP_template)
    n_ref = length(tx_refs);
    snr_list_db = NaN(1, n_ref);
    snr_sc_list = cell(1, n_ref);
    txafdem_list = cell(1, n_ref);
    ber_list = cell(1, n_ref);
    for tid = 1:n_ref
        tx_ref = normalize_tx_ref_shape_for_rx1_local(tx_refs{tid}, size(rx_sc,1));
        try
            [snr_frame_db, snr_sc_db, txafdem_matrix, ber_sc] = ...
                compute_rx1_style_snr_local(rx_sc, tx_ref, C, DSP_template);
            snr_list_db(tid) = snr_frame_db;
            snr_sc_list{tid} = snr_sc_db;
            txafdem_list{tid} = txafdem_matrix;
            ber_list{tid} = ber_sc;
        catch
            snr_list_db(tid) = NaN;
            snr_sc_list{tid} = NaN(size(rx_sc,1),1);
            txafdem_list{tid} = [];
            ber_list{tid} = [];
        end
    end
    [best_snr_db, best_tx_id] = max(snr_list_db);
    if isempty(best_snr_db) || ~isfinite(best_snr_db)
        best_tx_id = NaN; best_snr_db = NaN;
        best_snr_sc_db = NaN(size(rx_sc,1),1);
        best_txafdem = []; best_ber_sc = [];
    else
        best_snr_sc_db = snr_sc_list{best_tx_id};
        best_txafdem = txafdem_list{best_tx_id};
        best_ber_sc = ber_list{best_tx_id};
    end
end

function tx_ref_out = normalize_tx_ref_shape_for_rx1_local(tx_ref, n_sc)
    x = tx_ref;
    if size(x,1) == n_sc, tx_ref_out = x;
    elseif size(x,2) == n_sc, tx_ref_out = x.';
    else
        x = x(:); n_sym = floor(length(x)/n_sc);
        if n_sym < 1, error('Invalid shape'); end
        x = x(1:n_sc*n_sym); tx_ref_out = reshape(x, n_sc, n_sym);
    end
end

function rx_time = make_time_payload_local(rx_frame16, cfg)
    payload_start = cfg.header_len_16 + 1;
    rx = rx_frame16(:);
    if payload_start > length(rx), rx_time = zeros(cfg.M_time,1); return; end
    if payload_start + cfg.M_time - 1 <= length(rx)
        rx_time = rx(payload_start:payload_start+cfg.M_time-1);
    else
        rx_time = rx(payload_start:end);
        rx_time = [rx_time; zeros(cfg.M_time-length(rx_time),1)];
    end
    rx_time = rx_time - mean(rx_time);
    rx_time = rx_time ./ (rms(rx_time) + eps);
end

function CDM = make_cdm_local(rx_sc, nbin, clip_val)
    z = rx_sc(:);
    z = z(isfinite(real(z)) & isfinite(imag(z)));
    if isempty(z), CDM = zeros(nbin,nbin); return; end
    z = z - mean(z); z = z ./ (rms(abs(z)) + eps);
    zr = real(z); zi = imag(z);
    zr = max(min(zr,clip_val),-clip_val);
    zi = max(min(zi,clip_val),-clip_val);
    edges = linspace(-clip_val,clip_val,nbin+1);
    H = histcounts2(zi,zr,edges,edges);
    CDM = log1p(H);
    CDM = CDM ./ (max(CDM(:)) + eps);
end
