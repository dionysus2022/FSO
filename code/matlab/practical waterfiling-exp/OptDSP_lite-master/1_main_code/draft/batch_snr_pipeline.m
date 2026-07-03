%% batch_snr_pipeline.m
% 批量 SNR Pipeline
% 同步方式：iterative_rx1_style_pipeline.m（packet_edge_power_dect + LTS�?% 解调：demod_one_frame_from_lts_start
% SNR：三层结构（子载�?�?文件�?
clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

global PROG;
PROG.showMessagesLevel = 0;
initProg();

%% ===================== 配置 =====================

cfg = struct();
cfg.data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
cfg.rx_date   = '2026.06.26';
cfg.tx_root   = fullfile(cfg.data_root, 'tx_3frame_6mod');
cfg.out_root  = fullfile(cfg.data_root, 'dataset_batch_snr_v2');

cfg.Fs_rx   = 80e9;
cfg.Fs_base = 16e9;
cfg.n_frames = 3;

cfg.mod_names = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};
cfg.sub_list  = {'sub1','sub2','sub3'};

cfg.turb_map = containers.Map('KeyType','char','ValueType','char');
cfg.turb_map('sub1') = 'weak';
cfg.turb_map('sub2') = 'moderate';
cfg.turb_map('sub3') = 'strong';

% OFDM 参数
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

% 同步参数
cfg.sync_decim = 20;
cfg.num_corr_candidates = 60;
cfg.frame_margin_80 = 8000;
cfg.fine_search_len_80 = 30000;

%% ===================== 统计 =====================

all_frame_snr = [];
all_sc_snr = [];
all_file_snr = [];
all_mod_labels = {};
all_turb_labels = {};

snr_file_results = table();

stats = struct();
stats.total_files  = 0;
stats.total_frames = 0;
stats.ok_read    = 0;
stats.ok_sync    = 0;
stats.ok_demod   = 0;
stats.fail_read  = 0;
stats.fail_sync  = 0;
stats.fail_demod = 0;

%% ===================== 主循�?=====================

for mi = 1:length(cfg.mod_names)
    mod_name = cfg.mod_names{mi};
    [Mq, ~] = mod_to_order_bits_local(mod_name);

    nBpS_net = log2(Mq);
    if ~strcmp(mod_name, '32QAM')
        nBpS_net = nBpS_net - 0.2;
    end

    TX.SIG = setSignalParams('symRate', 8e9, 'M', Mq, ...
        'nPol', 1, 'nBpS', nBpS_net, 'nSyms', cfg.SIG.nSyms, ...
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

            % ---- 读取 RX .bin ----
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

            % ---- 重采样到 16G + wrap protect ----
            rx16 = resample(rx80, cfg.Fs_base, cfg.Fs_rx);
            rx16 = rx16 - mean(rx16);
            rx16 = rx16 ./ (mean(abs(rx16)) + eps);
            wrap_len = min(length(rx16), 3*cfg.frame_len_16);
            rx16_ext = [rx16, rx16(1:wrap_len)];

            % ---- 递归式单帧提取（iterative_rx1 方式�?---
            cursor = 1;
            file_frame_snr = [];
            file_sc_snr = [];

            for rk = 1:cfg.n_frames
                search_sig = rx16_ext(cursor:end);

                if length(search_sig) < cfg.frame_len_16, break; end

                try
                    [lts_start_rel, frame_start_rel, ~] = find_one_frame_start_rx1_style( ...
                        search_sig, cfg.ofdm, cfg.SIG.nSyms, cfg.zeros_head);
                catch
                    cursor = cursor + round(cfg.frame_len_16 * 0.5);
                    continue;
                end

                lts_start_abs   = cursor + lts_start_rel - 1;
                frame_start_abs = cursor + frame_start_rel - 1;

                % 解调
                try
                    [rx_sc, ~] = demod_one_frame_from_lts_start( ...
                        rx16_ext, lts_start_abs, cfg.ofdm, cfg.SIG.nSyms);
                catch
                    cursor = frame_start_abs + cfg.frame_len_16;
                    continue;
                end

                % 三层 SNR
                sc_snr = 10 * log10(mean(abs(rx_sc).^2, 2));
                frame_snr = 10 * log10(mean(abs(rx_sc(:)).^2));

                % TX 匹配
                [best_id, ~, ~] = compute_snr_three_level(rx_sc, tx_refs);

                file_frame_snr(end+1) = frame_snr;
                file_sc_snr = [file_sc_snr, sc_snr(:)];

                all_frame_snr(end+1) = frame_snr;
                all_sc_snr = [all_sc_snr, sc_snr(:)];
                all_mod_labels{end+1} = mod_name;
                all_turb_labels{end+1} = turb_name;

                stats.ok_demod = stats.ok_demod + 1;
                stats.total_frames = stats.total_frames + 1;

                cursor = frame_start_abs + cfg.frame_len_16;
            end

            stats.ok_sync = stats.ok_sync + 1;

            % 文件�?SNR
            if ~isempty(file_frame_snr)
                file_snr = 10 * log10(mean(10.^(file_frame_snr/10)));
                all_file_snr(end+1) = file_snr;

                snr_file_results = [snr_file_results; table(...
                    {mod_name}, {sub_name}, {turb_name}, sig_idx, ...
                    file_snr, mean(file_frame_snr), length(file_frame_snr), ...
                    'VariableNames', {'Mod','Sub','Turb','SigIdx', ...
                    'FileSNR_dB','MeanFrameSNR_dB','NFrames'})];
            end

            if mod(bi, 10) == 0 || bi == length(bin_list)
                ok = stats.ok_demod;
                fprintf('%s/%s: %d/%d OK, frames=%d (frSNR=%.1f)\n', ...
                    mod_name, sub_name, bi, length(bin_list), ok, ...
                    mean(file_frame_snr));
            end
        end
    end
end

%% ===================== 结果 =====================

fprintf('\n============================================\n');
fprintf('  Batch SNR Pipeline Complete\n');
fprintf('============================================\n');
fprintf('Files: %d OK, %d fail_read, %d fail_sync\n', ...
    stats.ok_read, stats.fail_read, stats.fail_sync);
fprintf('Frames demod OK: %d, fail: %d\n', ...
    stats.ok_demod, stats.fail_demod);

if isempty(all_frame_snr)
    fprintf('No successful frames\n');
    return;
end

fprintf('\n=== 3-Level SNR ===\n');
fprintf('Subcarrier SNR range: %.2f ~ %.2f dB\n', ...
    min(mean(all_sc_snr,2)), max(mean(all_sc_snr,2)));
fprintf('Frame SNR: mean=%.2f median=%.2f std=%.2f dB\n', ...
    mean(all_frame_snr), median(all_frame_snr), std(all_frame_snr));
fprintf('File SNR:  mean=%.2f median=%.2f std=%.2f dB\n', ...
    mean(all_file_snr), median(all_file_snr), std(all_file_snr));

%% ===================== 保存 =====================

out_dir = fullfile(cfg.out_root, 'snr_results');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

save(fullfile(out_dir, 'snr_summary.mat'), 'all_frame_snr', 'all_sc_snr', ...
    'all_file_snr', 'all_mod_labels', 'all_turb_labels');
writetable(snr_file_results, fullfile(out_dir, 'snr_per_file.csv'));

%% ===================== 直方�?=====================

figure('Position', [100 100 1400 900]);

subplot(2,3,1);
histogram(all_frame_snr, 50, 'FaceColor','#0072BD','EdgeColor','none');
xlabel('Frame SNR (dB)'); ylabel('Count');
title(sprintf('Frame SNR (n=%d, mean=%.2f)', length(all_frame_snr), mean(all_frame_snr))); grid on;

subplot(2,3,2); hold on;
mods = unique(all_mod_labels); colors = lines(length(mods));
for i = 1:length(mods)
    mask = strcmp(all_mod_labels, mods{i});
    histogram(all_frame_snr(mask), 30, 'FaceColor',colors(i,:), 'EdgeColor','none', 'FaceAlpha',0.5);
end
legend(mods, 'Location','northwest'); xlabel('Frame SNR (dB)'); title('SNR by Modulation'); grid on;

subplot(2,3,3);
histogram(all_file_snr, 40, 'FaceColor','#D95319','EdgeColor','none');
xlabel('File SNR (dB)'); ylabel('Count');
title(sprintf('File SNR (n=%d, mean=%.2f)', length(all_file_snr), mean(all_file_snr))); grid on;

subplot(2,3,4);
imagesc(all_sc_snr); colorbar; xlabel('Frame'); ylabel('Subcarrier');
title(sprintf('Subcarrier SNR (%d frames)', size(all_sc_snr,2)));
colormap('jet'); set(gca,'YDir','normal');

subplot(2,3,5);
ecdf(all_frame_snr); xlabel('Frame SNR (dB)'); ylabel('Cumulative (%)');
title('Cumulative SNR'); grid on; ylim([0 100]);
hold on;
for thresh = [0 5 10 15 20]
    pct = sum(all_frame_snr >= thresh)/length(all_frame_snr)*100;
    plot(thresh, 100-pct, 'ro');
    text(thresh+0.3, 100-pct+2, sprintf('%.0f%% > %ddB', 100-pct, thresh), 'FontSize',7);
end

subplot(2,3,6);
boxchart(categorical(all_mod_labels), all_frame_snr);
xlabel('Modulation'); ylabel('Frame SNR (dB)'); title('SNR Boxplot'); grid on;

saveas(gcf, fullfile(out_dir, 'snr_histogram.png'));
fprintf('Figure: %s\n', fullfile(out_dir, 'snr_histogram.png'));
fprintf('\n===== Done =====\n');

%% =====================================================================
%%                       辅助函数
%% =====================================================================

function [Mq,bits] = mod_to_order_bits_local(mod_name)
    switch mod_name
        case 'QPSK',    Mq=4; bits=2;
        case '16QAM',   Mq=16; bits=4;
        case '32QAM',   Mq=32; bits=5;
        case '64QAM',   Mq=64; bits=6;
        case '128QAM',  Mq=128; bits=7;
        case '256QAM',  Mq=256; bits=8;
    end
end

function y = read_keysight_bin_local(filename)
    fid = fopen(filename,'rb');
    if fid == -1, error('Cannot open: %s',filename); end
    fread(fid,2,'*char')';fread(fid,2,'*char')';
    fread(fid,1,'int32');fread(fid,1,'int32');
    fread(fid,1,'int32');fread(fid,1,'int32');
    fread(fid,1,'int32');num_points=fread(fid,1,'int32');
    fread(fid,1,'int32');fread(fid,1,'float32');
    fread(fid,1,'float64');fread(fid,1,'float64');
    fread(fid,1,'float64');fread(fid,1,'int32');
    fread(fid,1,'int32');fread(fid,16,'*char')';
    fread(fid,16,'*char')';fread(fid,24,'*char')';
    fread(fid,16,'*char')';fread(fid,1,'float64');
    fread(fid,1,'uint32');fread(fid,1,'int32');
    fread(fid,1,'int16');bpp=fread(fid,1,'int16');
    fread(fid,1,'int32');
    % ���������ݷǱ��� bpp
    if ~isscalar(bpp) || isempty(bpp)
        bpp = 2;
    else
        bpp = double(bpp);
    end
    switch bpp
        case 4, raw=fread(fid,num_points,'float32');
        case 2, raw=fread(fid,num_points,'int16');
        case 1, raw=fread(fid,num_points,'int8');
        otherwise, raw=fread(fid,num_points,'double');
    end
    fclose(fid);
    raw=raw(:); n=floor(length(raw)/2); raw=raw(1:2*n);
    y=double(raw(1:2:end))+1j*double(raw(2:2:end));
end

function x = load_ascii_complex_local(filename)
    tmp=load(filename);
    if size(tmp,2)>=2, x=complex(tmp(:,1),tmp(:,2));
    else, x=tmp(:); end
end

%% ===== rx1 风格同步（iterative_rx1 使用的）=====
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

    load('LongTrainSym_ini.mat','LongTrainSym_ini');
    LTS_f = LongTrainSym_ini(1:n_fft); LTS_f([1 n_fft/2+1]) = 0;
    ltrs_in = LTS_f; ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));

    [fine_time_est, data_df, max_peak_long] = rx_fine_time_sync_cross_corr( ...
        detected_packet, n_guard, ltrs_in, zeros_head, 0);

    lts_start = edge_index + fine_time_est - 1;
    frame_start = lts_start - (zeros_head + n_guard);
    if frame_start < 1, frame_start = 1; end
    if lts_start < 1 || lts_start + 2*n_fft - 1 > length(rx)
        error('LTS out of range');
    end
    info = struct();
    info.edge_index = edge_index;
    info.fine_time_est = fine_time_est;
    info.lts_start = lts_start;
    info.frame_start = frame_start;
    info.data_df = data_df;
    info.max_peak_long = max_peak_long;
    info.symbol_bits = symbol_bits;
end

%% ===== �?LTS 解调=====
function [rx_sc, info] = demod_one_frame_from_lts_start(rx, lts_start, ofdm, n_syms)
    rx = rx(:).';
    n_fft   = ofdm.NumberOfIFFTSamples;
    n_guard = ofdm.NumberOfGuardTime;
    sym_len = n_fft + n_guard;
    carrier_loc = 4:126;

    load('LongTrainSym_ini.mat','LongTrainSym_ini');
    LTS_f = LongTrainSym_ini(1:n_fft); LTS_f([1 n_fft/2+1]) = 0;
    ltrs_in = LTS_f; ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));
    LTS_f_ref = ltrs_in;

    lts1_end = lts_start + n_fft - 1;
    lts2_end = lts_start + 2*n_fft - 1;
    if lts_start < 1 || lts2_end > length(rx), error('frame too short for LTS'); end

    lts1 = rx(lts_start:lts1_end);
    lts2 = rx(lts_start+n_fft:lts2_end);

    pd = angle(sum(lts1(:).*conj(lts2(:))));
    cfo = pd/(2*pi*n_fft);
    n = 0:length(rx)-1;
    rx_cfo = rx .* exp(-1j*2*pi*cfo*n/n_fft);

    lts1 = rx_cfo(lts_start:lts1_end);
    lts2 = rx_cfo(lts_start+n_fft:lts2_end);

    data_start = lts_start + 2*n_fft;
    data_end = data_start + sym_len*n_syms - 1;

    if data_end > length(rx_cfo)
        remain = length(rx_cfo) - data_start + 1;
        nd = floor(remain/sym_len);
        if nd <= 0, error('no complete symbols'); end
        n_use = min(nd, n_syms);
    else
        n_use = n_syms;
    end

    data_end = data_start + sym_len*n_use - 1;
    dp = rx_cfo(data_start:data_end);

    dm = reshape(dp, sym_len, n_use);
    dn = dm(n_guard+1:end, :);
    fd = fft(dn, n_fft, 1) ./ sqrt(n_fft);

    lts_avg = (lts1(:)+lts2(:))/2;
    lts_fd = fft(lts_avg, n_fft) ./ sqrt(n_fft);
    H = lts_fd ./ (LTS_f_ref(:)+1e-12);
    H(abs(LTS_f_ref(:))<0.5) = 1;
    feq = fd ./ H;

    rx_sc = feq(carrier_loc, :);
    info = struct();
    info.cfo = cfo;
    info.n_use = n_use;
end

%% ===== 三层 SNR =====
function [best_id, frame_snr, sc_snr] = compute_snr_three_level(rx_sc, tx_refs)
    n_ref = length(tx_refs);
    sc_snr = 10 * log10(mean(abs(rx_sc).^2, 2));
    frame_snr = 10 * log10(mean(abs(rx_sc(:)).^2));

    mse_list = NaN(1, n_ref);
    for j = 1:n_ref
        tx_ref = tx_refs{j};
        n_sym = min(size(rx_sc,2), size(tx_ref,2));
        rx_use = rx_sc(:, 1:n_sym);
        tx_use = tx_ref(:, 1:n_sym);
        rx_norm = rx_use ./ (norm(rx_use,'fro')+eps);
        tx_norm = tx_use ./ (norm(tx_use,'fro')+eps);
        mse_list(j) = mean(abs(rx_norm(:)-tx_norm(:)).^2);
    end
    [~, best_id] = min(mse_list);
end
