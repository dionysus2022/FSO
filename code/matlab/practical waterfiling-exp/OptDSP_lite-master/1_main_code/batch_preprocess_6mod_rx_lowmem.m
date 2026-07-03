%% batch_preprocess_6mod_rx_lowmem.m
% =========================================================
% 六种调制格式真实实验 RX 低内存预处理脚本
%
% 适用场景：
%   MATLAB 内存只有 16GB 左右，parfor 版本容易崩溃。
%
% 低内存设计：
%   1) 不使用 parfor，逐文件处理，避免多个 worker 同时读入大 bin；
%   2) 不把所有 results 存在 cell 里，边处理边写 manifest；
%   3) 默认不保存 rx_time / rx_frame16_lts / sync_info / demod_info 等大型或结构字段；
%   4) 默认不计算 TX 匹配、symDemapper、EVM、BER、子载波 SNR；
%   5) 每处理完一个 bin 立即 clear 临时变量；
%   6) 只保存识别模型真正需要的紧凑样本：
%        rx_sc, Y_iq, cdm64, blind_stats, mod_label, turb_label。
%
% 输出：
%   preprocessed_uniform_qam_rx_lowmem/2026.06.28/
%       QPSK/weak/*.mat
%       QPSK/strong/*.mat
%       ...
%       manifest_all.csv
%       run_summary_lowmem.txt
%
% 重要：
%   如果仍然内存不足，请把 cfg.save_Y_iq = false，只保存 rx_sc、cdm64、blind_stats。
% =========================================================

clear; clear global; close all; clc;

%% ===================== 基础路径 =====================
project_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master';
data_root    = fullfile(project_root, '2_Data_Results');
addpath(genpath(project_root));

global PROG;
PROG.showMessagesLevel = 0;
try
    initProg();
catch
    warning('initProg() failed or not found. Continue anyway.');
end

%% ===================== 用户配置 =====================
cfg = struct();
cfg.project_root = project_root;
cfg.data_root    = data_root;
cfg.date_tag     = '2026.06.28';

cfg.mod_list       = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};
cfg.mod_label_list = [0, 1, 2, 3, 4, 5];

cfg.turb_subdirs = {'sub01','sub03'};
cfg.turb_names   = {'weak','strong'};
cfg.turb_labels  = [0, 1];

cfg.out_root_all = fullfile(data_root, 'preprocessed_uniform_qam_rx_lowmem', cfg.date_tag);
if ~exist(cfg.out_root_all, 'dir'), mkdir(cfg.out_root_all); end

% 采样率
cfg.Fs_rx   = 80e9;
cfg.Fs_base = 16e9;

% 每个 bin 默认 3 帧
cfg.n_frames_per_file = 3;

% OFDM 参数
cfg.zeros_head = 80;
cfg.n_fft      = 256;
cfg.n_guard    = 16;
cfg.n_syms     = 128;
cfg.carrier_loc = 4:126;
cfg.n_sc = length(cfg.carrier_loc);
cfg.sym_len = cfg.n_fft + cfg.n_guard;
cfg.frame_len_16 = cfg.zeros_head + cfg.n_guard + 2*cfg.n_fft + cfg.sym_len*cfg.n_syms;
cfg.frame_pre_lts = cfg.zeros_head + cfg.n_guard - 5;
cfg.next_search_backoff = 800;

% CDM 参数
cfg.cdm_bins = 64;
cfg.cdm_clip = 3.0;

% 低内存保存选项
cfg.save_rx_sc = true;
cfg.save_Y_iq  = true;      % 如果仍崩溃，改成 false；Python 可由 rx_sc 再生成 Y_iq
cfg.save_cdm64 = true;
cfg.save_blind_stats = true;

% 默认不要保存这些大字段
cfg.save_rx_time = false;
cfg.save_rx_frame16_lts = false;
cfg.save_sync_info = false;
cfg.save_demod_info = false;

% 不画图
cfg.plot_quality_check = false;

% 不使用 GPU，不使用 parfor
cfg.use_cuda = false;
cfg.use_parallel_cpu = false;

% 正式运行用 inf；调试可设为 12
cfg.debug_max_jobs = inf;

% 文件筛选
cfg.max_files_per_turb_per_mod = inf;

% 如果某帧失败，继续处理下一帧
cfg.continue_on_frame_error = true;

% 每处理多少个文件打印一次进度
cfg.progress_every_n_files = 5;

%% ===================== LTS =====================
LTS = make_lts_local(cfg.n_fft);
LTS_freq = LTS.freq; %#ok<NASGU>

%% ===================== 发现任务 =====================
all_jobs = discover_all_jobs_6mod_local(cfg);

if isempty(all_jobs)
    error('No RX .bin files found. Please check rx_data paths.');
end

if isfinite(cfg.debug_max_jobs)
    all_jobs = all_jobs(1:min(numel(all_jobs), cfg.debug_max_jobs));
end

fprintf('\n============================================\n');
fprintf('Six-modulation LOW-MEM preprocessing\n');
fprintf('============================================\n');
fprintf('Output root    : %s\n', cfg.out_root_all);
fprintf('Total RX files : %d\n', numel(all_jobs));
fprintf('Expected frames: %d\n', numel(all_jobs)*cfg.n_frames_per_file);
fprintf('Parallel CPU   : false\n');
fprintf('Save Y_iq      : %d\n', cfg.save_Y_iq);
fprintf('Save rx_time   : %d\n', cfg.save_rx_time);
fprintf('============================================\n');

%% ===================== Manifest =====================
manifest_file = fullfile(cfg.out_root_all, 'manifest_all.csv');
fid_manifest = fopen(manifest_file, 'w');
if fid_manifest == -1
    error('Cannot create manifest: %s', manifest_file);
end
write_manifest_header_lowmem(fid_manifest);

%% ===================== 逐文件处理 =====================
t_all = tic;

processed_files = 0;
failed_files = 0;
processed_frames = 0;
failed_frames = 0;
global_frame_id = 0;

for j = 1:numel(all_jobs)
    job = all_jobs(j);

    fprintf('\n[%d/%d] %s | %s | %s\n', ...
        j, numel(all_jobs), job.mod_name, job.turb_name, job.rx_name);

    try
        [file_status, frame_records, n_ok, n_fail] = process_one_rx_file_lowmem(job, cfg, LTS);

        if strcmp(file_status, 'ok')
            processed_files = processed_files + 1;
        else
            failed_files = failed_files + 1;
        end

        processed_frames = processed_frames + n_ok;
        failed_frames = failed_frames + n_fail;

        for r = 1:numel(frame_records)
            rec = frame_records(r);
            if strcmp(rec.status, 'ok')
                global_frame_id = global_frame_id + 1;
                rec.global_frame_id = global_frame_id;
            else
                rec.global_frame_id = NaN;
            end
            write_manifest_row_lowmem(fid_manifest, rec);
        end

    catch ME
        failed_files = failed_files + 1;
        failed_frames = failed_frames + cfg.n_frames_per_file;

        rec = make_manifest_record_lowmem(job, cfg, NaN, '', ...
            'failed_file', ME.message, NaN, NaN, NaN, NaN, NaN(1,16));
        write_manifest_row_lowmem(fid_manifest, rec);

        fprintf('  FAILED FILE: %s\n', ME.message);
    end

    if mod(j, cfg.progress_every_n_files) == 0 || j == numel(all_jobs)
        elapsed = toc(t_all);
        avg_file = elapsed / j;
        remain = avg_file * (numel(all_jobs)-j);

        fprintf('\n[PROGRESS] %d/%d files | OK frames=%d | Failed frames=%d | elapsed %.1f min | remaining %.1f min\n', ...
            j, numel(all_jobs), processed_frames, failed_frames, elapsed/60, remain/60);
    end

    % 强制清理图形和 Java 临时资源
    close all hidden;
    drawnow;
end

fclose(fid_manifest);

total_elapsed = toc(t_all);

%% ===================== 保存汇总 =====================
run_summary = struct();
run_summary.cfg = cfg;
run_summary.n_jobs = numel(all_jobs);
run_summary.processed_files = processed_files;
run_summary.failed_files = failed_files;
run_summary.processed_frames = processed_frames;
run_summary.failed_frames = failed_frames;
run_summary.actual_total_time_sec = total_elapsed;
run_summary.manifest_file = manifest_file;
run_summary.out_root_all = cfg.out_root_all;

save(fullfile(cfg.out_root_all, 'run_summary_lowmem.mat'), 'run_summary', '-v7.3');
write_run_summary_txt_lowmem(run_summary, fullfile(cfg.out_root_all, 'run_summary_lowmem.txt'));

fprintf('\n============================================\n');
fprintf(' LOW-MEM preprocessing completed\n');
fprintf('============================================\n');
fprintf('Processed files : %d\n', processed_files);
fprintf('Failed files    : %d\n', failed_files);
fprintf('Processed frames: %d\n', processed_frames);
fprintf('Failed frames   : %d\n', failed_frames);
fprintf('Actual time     : %.1f min = %.2f h\n', total_elapsed/60, total_elapsed/3600);
fprintf('Manifest        : %s\n', manifest_file);
fprintf('Out root        : %s\n', cfg.out_root_all);
fprintf('============================================\n');

%% =====================================================================
%% Job discovery
%% =====================================================================
function all_jobs = discover_all_jobs_6mod_local(cfg)
    all_jobs = struct([]);
    job_id = 0;

    for mi = 1:numel(cfg.mod_list)
        mod_name = cfg.mod_list{mi};
        mod_label = cfg.mod_label_list(mi);

        for ti = 1:numel(cfg.turb_subdirs)
            turb_subdir = cfg.turb_subdirs{ti};
            turb_name   = cfg.turb_names{ti};
            turb_label  = cfg.turb_labels(ti);

            rx_dir = fullfile(cfg.data_root, 'rx_data', cfg.date_tag, mod_name, turb_subdir);

            if ~exist(rx_dir, 'dir')
                warning('RX dir not found: %s', rx_dir);
                continue;
            end

            files = dir(fullfile(rx_dir, '*.bin'));
            files = sort_files_by_number_local(files);

            if isfinite(cfg.max_files_per_turb_per_mod)
                files = files(1:min(numel(files), cfg.max_files_per_turb_per_mod));
            end

            fprintf('[DISCOVER] %-7s | %-6s | %d files | %s\n', ...
                mod_name, turb_name, numel(files), rx_dir);

            for i = 1:numel(files)
                job_id = job_id + 1;
                all_jobs(job_id).job_id = job_id; %#ok<AGROW>
                all_jobs(job_id).mod_name = mod_name;
                all_jobs(job_id).mod_label = mod_label;
                all_jobs(job_id).turb_subdir = turb_subdir;
                all_jobs(job_id).turb_name = turb_name;
                all_jobs(job_id).turb_label = turb_label;
                all_jobs(job_id).rx_dir = rx_dir;
                all_jobs(job_id).rx_file = fullfile(files(i).folder, files(i).name);
                all_jobs(job_id).rx_name = files(i).name;
                all_jobs(job_id).sig_idx = infer_sig_idx_from_filename_local(files(i).name, i);
            end
        end
    end
end

%% =====================================================================
%% One file processing
%% =====================================================================
function [file_status, frame_records, n_ok, n_fail] = process_one_rx_file_lowmem(job, cfg, LTS)

    file_status = 'failed';
    frame_records = struct([]);
    n_ok = 0;
    n_fail = 0;

    set_out_dir = fullfile(cfg.out_root_all, job.mod_name, job.turb_name);
    if ~exist(set_out_dir, 'dir'), mkdir(set_out_dir); end

    file_base = erase(job.rx_name, '.bin');
    safe_base = regexprep(file_base, '[^\w\-]', '_');
    file_out_dir = fullfile(set_out_dir, sprintf('%s_sig%04d_%s', job.mod_name, job.sig_idx, safe_base));
    if ~exist(file_out_dir, 'dir'), mkdir(file_out_dir); end

    try
        [rx80, ~] = read_keysight_bin_robust_real_local(job.rx_file);
        rx80 = rx80(:).';
        rx80 = rx80 - mean(rx80);
        rx80 = rx80 ./ (rms(rx80) + eps);

        % 降采样到 16G
        rx16 = resample(rx80, cfg.Fs_base, cfg.Fs_rx);

        % 读完和重采样后立刻释放原始大数组
        clear rx80;

        rx16 = rx16(:).';
        rx16 = rx16 - mean(rx16);
        rx16 = rx16 ./ (mean(abs(rx16)) + eps);

        wrap_len = min(length(rx16), 3*cfg.frame_len_16);
        rx16_ext = [rx16, rx16(1:wrap_len)];

        % 释放 rx16，只保留扩展后的数组
        clear rx16;

        cursor = 1;

        for rk = 1:cfg.n_frames_per_file
            try
                if cursor >= length(rx16_ext) - cfg.frame_len_16
                    error('cursor too close to end');
                end

                search_sig = rx16_ext(cursor:end);
                [lts_start_rel, frame_start_rel, sync_info] = find_one_frame_start_rx1_style_local(search_sig, cfg);
                clear search_sig;

                lts_start_abs   = cursor + lts_start_rel - 1;
                frame_start_abs = cursor + frame_start_rel - 1;

                [rx_sc, demod_info] = demod_one_frame_from_lts_start_lowmem(rx16_ext, lts_start_abs, LTS, cfg);

                cdm64 = make_cdm_from_rxsc_local(rx_sc, cfg.cdm_bins, cfg.cdm_clip);
                [blind_stats, blind_stats_names] = make_blind_stats_local(rx_sc);

                frame_mat = fullfile(file_out_dir, sprintf('frame_%02d.mat', rk));

                sample = struct();

                if cfg.save_rx_sc
                    sample.rx_sc = single(rx_sc);
                end

                if cfg.save_Y_iq
                    sample.Y_iq = make_iq_tensor_local(rx_sc);
                end

                if cfg.save_cdm64
                    sample.cdm64 = single(cdm64);
                end

                if cfg.save_blind_stats
                    sample.blind_stats = single(blind_stats(:).');
                    sample.blind_stats_names = blind_stats_names;
                end

                sample.mod_name = job.mod_name;
                sample.mod_label = int32(job.mod_label);
                sample.turb_name = job.turb_name;
                sample.turb_label = int32(job.turb_label);
                sample.turb_subdir = job.turb_subdir;
                sample.rx_file = job.rx_file;
                sample.rx_name = job.rx_name;
                sample.sig_idx = int32(job.sig_idx);
                sample.rx_frame_idx = int32(rk);
                sample.lts_start_abs = int64(lts_start_abs);
                sample.frame_start_abs = int64(frame_start_abs);
                sample.fast_mode = true;
                sample.lowmem_mode = true;

                % 兼容字段：快速低内存模式不计算这些
                sample.snr_sc_db = single(NaN(cfg.n_sc, 1));
                sample.snr_frame_db = single(NaN);

                if cfg.save_sync_info
                    sample.sync_info = sync_info;
                end
                if cfg.save_demod_info
                    sample.demod_info = demod_info;
                end

                save(frame_mat, '-struct', 'sample', '-v7');

                n_ok = n_ok + 1;

                rec = make_manifest_record_lowmem(job, cfg, rk, frame_mat, ...
                    'ok', '', lts_start_abs, frame_start_abs, demod_info.cfo, demod_info.n_use, blind_stats);
                frame_records = append_record_local(frame_records, rec);

                cursor_next = frame_start_abs + cfg.frame_len_16 - cfg.next_search_backoff;
                if cursor_next <= cursor
                    cursor_next = cursor + round(0.8 * cfg.frame_len_16);
                end
                cursor = cursor_next;

                clear rx_sc cdm64 blind_stats sample sync_info demod_info;

            catch ME_frame
                n_fail = n_fail + 1;

                rec = make_manifest_record_lowmem(job, cfg, rk, '', ...
                    'failed', ME_frame.message, NaN, NaN, NaN, NaN, NaN(1,16));
                frame_records = append_record_local(frame_records, rec);

                fprintf('  Frame %d failed: %s\n', rk, ME_frame.message);

                if cfg.continue_on_frame_error
                    cursor = cursor + round(0.8 * cfg.frame_len_16);
                else
                    rethrow(ME_frame);
                end
            end
        end

        clear rx16_ext;

        file_status = 'ok';

    catch ME_file
        n_fail = cfg.n_frames_per_file;

        rec = make_manifest_record_lowmem(job, cfg, NaN, '', ...
            'failed_file', ME_file.message, NaN, NaN, NaN, NaN, NaN(1,16));
        frame_records = append_record_local(frame_records, rec);

        fprintf('  File failed: %s\n', ME_file.message);
    end
end

function arr = append_record_local(arr, rec)
    if isempty(arr)
        arr = rec;
    else
        arr(end+1) = rec;
    end
end

%% =====================================================================
%% Sync / demod low memory
%% =====================================================================
function [lts_start, frame_start, info] = find_one_frame_start_rx1_style_local(rx, cfg)
    rx = rx(:).';
    n_fft = cfg.n_fft;
    n_guard = cfg.n_guard;

    symbol_bits = cfg.zeros_head + n_guard + 2*n_fft + (n_fft + n_guard) * cfg.n_syms;
    search_len = min(length(rx), 2 * symbol_bits);

    if search_len < symbol_bits
        error('input too short for sync: len=%d', length(rx));
    end

    search_sig = rx(1:search_len);

    [detected_packet, edge_index] = packet_edge_power_dect(search_sig, cfg.zeros_head);

    load('LongTrainSym_ini.mat', 'LongTrainSym_ini');
    LTS_f = LongTrainSym_ini(1:n_fft);
    LTS_f([1 n_fft/2+1]) = 0;
    ltrs_in = LTS_f;
    ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));

    [fine_time_est, data_df, max_peak_long] = rx_fine_time_sync_cross_corr( ...
        detected_packet, n_guard, ltrs_in, cfg.zeros_head, 0);

    lts_start = edge_index + fine_time_est - 1;
    frame_start = lts_start - cfg.frame_pre_lts;
    if frame_start < 1
        frame_start = 1;
    end

    info = struct();
    info.edge_index = edge_index;
    info.fine_time_est = fine_time_est;
    info.lts_start = lts_start;
    info.frame_start = frame_start;
    info.data_df = data_df;
    info.max_peak_long = max_peak_long;
end

function [rx_sc, info] = demod_one_frame_from_lts_start_lowmem(rx, lts_start, LTS, cfg)
    rx = rx(:).';
    n_fft = cfg.n_fft;
    n_guard = cfg.n_guard;
    sym_len = cfg.sym_len;

    lts1_start = lts_start;
    lts1_end   = lts_start + n_fft - 1;
    lts2_start = lts_start + n_fft;
    lts2_end   = lts_start + 2*n_fft - 1;

    if lts_start < 1 || lts2_end > length(rx)
        error('frame too short for LTS: remaining=%d', length(rx)-lts_start+1);
    end

    lts1 = rx(lts1_start:lts1_end);
    lts2 = rx(lts2_start:lts2_end);

    cfo = angle(sum(lts1(:).*conj(lts2(:)))) / (2*pi*n_fft);

    remaining = length(rx) - lts_start + 1;
    n = 0:remaining-1;

    % 这里 rx_comp 是单帧后续片段，处理完立刻释放
    rx_comp = rx(lts_start:end) .* exp(-1j*2*pi*cfo*n/n_fft);

    lts1c = rx_comp(1:n_fft);
    lts2c = rx_comp(n_fft+1:2*n_fft);

    data_start = 2*n_fft + 1;
    dp_all = rx_comp(data_start:end);

    nd = floor(length(dp_all) / sym_len);
    if nd <= 0
        error('no complete OFDM symbols: nd=0');
    end

    n_use = min(nd, cfg.n_syms);
    dp = dp_all(1:n_use * sym_len);
    dm = reshape(dp, sym_len, n_use);
    dn = dm(n_guard+1:end, :);

    fd = fft(dn, n_fft, 1) / sqrt(n_fft);

    lts_avg = (lts1c(:) + lts2c(:)) / 2;
    lts_fd = fft(lts_avg, n_fft) / sqrt(n_fft);

    H = lts_fd ./ (LTS.freq(:) + 1e-12);
    H(abs(LTS.freq(:)) < 0.5) = 1;

    feq = fd ./ H;
    rx_sc = feq(cfg.carrier_loc, :);

    info = struct();
    info.cfo = cfo;
    info.n_use = n_use;
    info.nd_available = nd;

    clear rx_comp dp_all dp dm dn fd feq;
end

%% =====================================================================
%% Features
%% =====================================================================
function Y_iq = make_iq_tensor_local(rx_sc)
    Y_iq = zeros(2, size(rx_sc,1), size(rx_sc,2), 'single');
    Y_iq(1,:,:) = single(real(rx_sc));
    Y_iq(2,:,:) = single(imag(rx_sc));
end

function cdm = make_cdm_from_rxsc_local(rx_sc, nbin, clip_val)
    z = rx_sc(:);
    z = z(isfinite(real(z)) & isfinite(imag(z)));

    if isempty(z)
        cdm = zeros(nbin, nbin, 'single');
        return;
    end

    z = z - mean(z);
    z = z ./ (rms(abs(z)) + eps);

    zr = max(min(real(z), clip_val), -clip_val);
    zi = max(min(imag(z), clip_val), -clip_val);

    edges = linspace(-clip_val, clip_val, nbin + 1);
    H = histcounts2(zi, zr, edges, edges);

    cdm = log1p(H);
    cdm = cdm ./ (max(cdm(:)) + eps);
    cdm = single(cdm);
end

function [stats, names] = make_blind_stats_local(rx_sc)
    z = rx_sc(:);
    z = z(isfinite(real(z)) & isfinite(imag(z)));

    names = { ...
        'amp_mean','amp_std','amp_skew','amp_kurt', ...
        'papr_db','i_mean','i_std','i_skew','i_kurt', ...
        'q_mean','q_std','q_skew','q_kurt', ...
        'phase_diff_std','phase_concentration','iq_corr'};

    if isempty(z)
        stats = NaN(1, numel(names));
        return;
    end

    z = z - mean(z);
    z = z ./ sqrt(mean(abs(z).^2) + eps);

    amp = abs(z);
    ii = real(z);
    qq = imag(z);
    ph = unwrap(angle(z));
    dph = diff(ph);

    amp_mean = mean(amp);
    amp_std  = std(amp);
    amp_skew = skewness_manual_local(amp);
    amp_kurt = kurtosis_manual_local(amp);

    papr_db = 10*log10(max(abs(z).^2) / (mean(abs(z).^2) + eps));

    i_mean = mean(ii);
    i_std  = std(ii);
    i_skew = skewness_manual_local(ii);
    i_kurt = kurtosis_manual_local(ii);

    q_mean = mean(qq);
    q_std  = std(qq);
    q_skew = skewness_manual_local(qq);
    q_kurt = kurtosis_manual_local(qq);

    if isempty(dph)
        phase_diff_std = NaN;
    else
        phase_diff_std = std(dph);
    end

    phase_concentration = abs(mean(exp(1j*angle(z))));

    if std(ii) < eps || std(qq) < eps
        iq_corr = 0;
    else
        C = corrcoef(ii, qq);
        iq_corr = C(1,2);
    end

    stats = [ ...
        amp_mean, amp_std, amp_skew, amp_kurt, ...
        papr_db, i_mean, i_std, i_skew, i_kurt, ...
        q_mean, q_std, q_skew, q_kurt, ...
        phase_diff_std, phase_concentration, iq_corr];
end

function s = skewness_manual_local(x)
    x = x(:);
    x = x(isfinite(x));
    if numel(x) < 3
        s = NaN;
        return;
    end
    mu = mean(x);
    sd = std(x) + eps;
    s = mean(((x - mu)./sd).^3);
end

function k = kurtosis_manual_local(x)
    x = x(:);
    x = x(isfinite(x));
    if numel(x) < 4
        k = NaN;
        return;
    end
    mu = mean(x);
    sd = std(x) + eps;
    k = mean(((x - mu)./sd).^4);
end

%% =====================================================================
%% Manifest
%% =====================================================================
function rec = make_manifest_record_lowmem(job, cfg, rx_frame_idx, out_mat, ...
    status, message, lts_start_abs, frame_start_abs, cfo, n_use, blind_stats)

    rec = struct();
    rec.global_frame_id = NaN;
    rec.status = status;
    rec.message = sanitize_csv_text_local(message);
    rec.mod_name = job.mod_name;
    rec.mod_label = job.mod_label;
    rec.turb_name = job.turb_name;
    rec.turb_label = job.turb_label;
    rec.turb_subdir = job.turb_subdir;
    rec.rx_file = job.rx_file;
    rec.rx_name = job.rx_name;
    rec.sig_idx = job.sig_idx;
    rec.rx_frame_idx = rx_frame_idx;
    rec.lts_start_abs = lts_start_abs;
    rec.frame_start_abs = frame_start_abs;
    rec.cfo = cfo;
    rec.n_use = n_use;

    if nargin < 11 || isempty(blind_stats) || all(isnan(blind_stats(:)))
        blind_stats = NaN(1,16);
    end

    blind_stats = double(blind_stats(:).');
    if numel(blind_stats) < 16
        blind_stats = [blind_stats, NaN(1,16-numel(blind_stats))];
    end

    rec.blind_stats = blind_stats(1:16);
    rec.out_mat = out_mat;
end

function write_manifest_header_lowmem(fid)
    fprintf(fid, ['global_frame_id,status,message,mod_name,mod_label,turb_name,turb_label,turb_subdir,' ...
        'rx_file,rx_name,sig_idx,rx_frame_idx,lts_start_abs,frame_start_abs,cfo,n_use,' ...
        'amp_mean,amp_std,amp_skew,amp_kurt,papr_db,' ...
        'i_mean,i_std,i_skew,i_kurt,q_mean,q_std,q_skew,q_kurt,' ...
        'phase_diff_std,phase_concentration,iq_corr,out_mat\n']);
end

function write_manifest_row_lowmem(fid, rec)
    bs = rec.blind_stats;
    fprintf(fid, ['%g,%s,%s,%s,%d,%s,%d,%s,%s,%s,%d,%g,%g,%g,%.12g,%g,' ...
        '%.12g,%.12g,%.12g,%.12g,%.12g,' ...
        '%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,' ...
        '%.12g,%.12g,%.12g,%s\n'], ...
        rec.global_frame_id, rec.status, sanitize_csv_text_local(rec.message), ...
        rec.mod_name, rec.mod_label, rec.turb_name, rec.turb_label, rec.turb_subdir, ...
        sanitize_csv_text_local(rec.rx_file), sanitize_csv_text_local(rec.rx_name), ...
        rec.sig_idx, rec.rx_frame_idx, rec.lts_start_abs, rec.frame_start_abs, rec.cfo, rec.n_use, ...
        bs(1), bs(2), bs(3), bs(4), bs(5), ...
        bs(6), bs(7), bs(8), bs(9), bs(10), bs(11), bs(12), bs(13), ...
        bs(14), bs(15), bs(16), sanitize_csv_text_local(rec.out_mat));
end

function t = sanitize_csv_text_local(t)
    if isempty(t)
        t = '';
        return;
    end
    if isstring(t), t = char(t); end
    if isnumeric(t), t = num2str(t); end
    t = strrep(t, ',', ';');
    t = strrep(t, newline, ' ');
    t = strrep(t, sprintf('\r'), ' ');
    t = strrep(t, sprintf('\n'), ' ');
end

function write_run_summary_txt_lowmem(run_summary, filename)
    fid = fopen(filename, 'w');
    fprintf(fid, 'Processed files : %d\n', run_summary.processed_files);
    fprintf(fid, 'Failed files    : %d\n', run_summary.failed_files);
    fprintf(fid, 'Processed frames: %d\n', run_summary.processed_frames);
    fprintf(fid, 'Failed frames   : %d\n', run_summary.failed_frames);
    fprintf(fid, 'Actual total time: %.4f sec = %.4f min = %.4f h\n', ...
        run_summary.actual_total_time_sec, run_summary.actual_total_time_sec/60, run_summary.actual_total_time_sec/3600);
    fprintf(fid, 'Manifest: %s\n', run_summary.manifest_file);
    fprintf(fid, 'Output root: %s\n', run_summary.out_root_all);
    fprintf(fid, 'Save Y_iq: %d\n', run_summary.cfg.save_Y_iq);
    fprintf(fid, 'Save rx_time: %d\n', run_summary.cfg.save_rx_time);
    fclose(fid);
end

%% =====================================================================
%% Helpers
%% =====================================================================
function LTS = make_lts_local(n_fft)
    load('LongTrainSym_ini.mat', 'LongTrainSym_ini');
    LTS_f0 = LongTrainSym_ini(1:n_fft);
    LTS_f0([1 n_fft/2+1]) = 0;
    ltrs_in = LTS_f0;
    ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));
    LTS.freq = ltrs_in(:);
    LTS.time = ifft(ltrs_in(:));
end

function files = sort_files_by_number_local(files)
    if isempty(files), return; end
    nums = zeros(numel(files),1);
    for i = 1:numel(files)
        nums(i) = infer_sig_idx_from_filename_local(files(i).name, i);
    end
    [~, idx] = sortrows([nums(:), (1:numel(files)).']);
    files = files(idx);
end

function sig_idx = infer_sig_idx_from_filename_local(name, fallback)
    tok = regexp(name, '(\d+)', 'tokens', 'once');
    if isempty(tok)
        sig_idx = fallback;
    else
        sig_idx = str2double(tok{1});
        if isnan(sig_idx), sig_idx = fallback; end
    end
end

%% =====================================================================
%% Robust Keysight real reader
%% =====================================================================
function [y, info] = read_keysight_bin_robust_real_local(filename)
    info = struct();

    try
        [y, info] = read_keysight_real_standard_or_infer_local(filename, false);
        info.method = 'standard_bpp_real';
        return;
    catch ME1
        info.standard_error = ME1.message;
    end

    try
        [y, info] = read_keysight_real_standard_or_infer_local(filename, true);
        info.method = 'inferred_bpp_real';
        return;
    catch ME2
        info.infer_error = ME2.message;
    end

    error('read_keysight_bin_robust_real failed: standard=[%s], infer=[%s]', ...
        info.standard_error, info.infer_error);
end

function [y, info] = read_keysight_real_standard_or_infer_local(filename, force_infer)
    fid = fopen(filename, 'rb', 'ieee-le');
    if fid == -1
        error('Cannot open: %s', filename);
    end
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fread(fid, 2, '*char')';
    fread(fid, 2, '*char')';
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
    fread(fid, 16, '*char')';
    fread(fid, 16, '*char')';
    fread(fid, 24, '*char')';
    fread(fid, 16, '*char')';
    fread(fid, 1, 'float64');
    fread(fid, 1, 'uint32');
    fread(fid, 1, 'int32');
    fread(fid, 1, 'int16');
    bpp_read = fread(fid, 1, 'int16');
    buffer_size = fread(fid, 1, 'int32');

    data_start = ftell(fid);
    d = dir(filename);
    remain_bytes = d.bytes - data_start;

    if isempty(num_points) || numel(num_points) ~= 1 || num_points <= 0
        error('invalid num_points');
    end

    bpp_candidates = [];

    if ~force_infer
        if isempty(bpp_read) || numel(bpp_read) ~= 1 || ~ismember(double(bpp_read), [1 2 4 8])
            error('invalid bpp: %s', mat2str(bpp_read));
        end
        bpp_candidates = double(bpp_read);
    else
        if ~isempty(bpp_read) && numel(bpp_read) == 1 && ismember(double(bpp_read), [1 2 4 8])
            bpp_candidates(end+1) = double(bpp_read); %#ok<AGROW>
        end

        if ~isempty(buffer_size) && numel(buffer_size) == 1 && buffer_size > 0
            bpp_candidates(end+1) = round(double(buffer_size) / double(num_points)); %#ok<AGROW>
        end

        if remain_bytes > 0
            bpp_candidates(end+1) = round(double(remain_bytes) / double(num_points)); %#ok<AGROW>
        end

        bpp_candidates = unique([bpp_candidates 4 2 1 8], 'stable');
        bpp_candidates = bpp_candidates(ismember(bpp_candidates, [1 2 4 8]));
    end

    last_msg = '';
    for ii = 1:length(bpp_candidates)
        bpp = bpp_candidates(ii);
        try
            fseek(fid, data_start, 'bof');
            raw = read_raw_by_bpp_real_local(fid, double(num_points), bpp);
            raw = double(raw(:)).';

            if length(raw) < 1000
                error('raw too short');
            end
            if std(raw(1:min(5000,end))) == 0
                error('zero variance raw');
            end

            y = raw;

            info = struct();
            info.method = 'standard_or_infer_real';
            info.bpp = double(bpp_read);
            info.inferred_bpp = bpp;
            info.num_points = double(num_points);
            info.buffer_size = double(buffer_size);
            info.data_start = data_start;
            info.remain_bytes = remain_bytes;
            return;
        catch ME
            last_msg = ME.message;
        end
    end

    error('all bpp candidates failed: %s', last_msg);
end

function raw = read_raw_by_bpp_real_local(fid, num_points, bpp)
    switch bpp
        case 4
            raw = fread(fid, num_points, 'float32');
        case 2
            raw = fread(fid, num_points, 'int16');
        case 1
            raw = fread(fid, num_points, 'int8');
        case 8
            raw = fread(fid, num_points, 'double');
        otherwise
            error('unsupported bpp=%g', bpp);
    end
end
