%% batch_preprocess_6mod_rx_fast.m
% =========================================================
% 六种调制格式真实实验RX批量快速预处理脚本
% 适用目标：
%   真实实验湍流数据 + 多表征融合 + 小样本鲁棒识别
%
% 加速点：
%   1) 一次性处理 QPSK/16QAM/32QAM/64QAM/128QAM/256QAM；
%   2) 默认不做逐子载波 symDemapper/EVM/SNR 匹配；
%   3) 默认不保存大型 file_result.mat；
%   4) 默认不画质检图；
%   5) 支持按 RX 文件级 parfor 并行；
%   6) 只保存识别真正需要的变量：
%        rx_sc, Y_iq, cdm64, blind_stats, mod_label, turb_label 等。
%
% 重要说明：
%   - 本脚本继续使用原同步/deOFDM/CFO补偿/均衡流程；
%   - 输出样本用于 Python 端多表征融合识别；
%   - SNR/EVM 建议另开慢速质检脚本抽样计算，不建议在主预处理中全量计算。
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

% 六种调制格式与标签，必须固定，后续Python训练也按这个顺序
cfg.mod_list       = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};
cfg.mod_label_list = [0, 1, 2, 3, 4, 5];

% 两种湍流
cfg.turb_subdirs = {'sub01','sub03'};
cfg.turb_names   = {'weak','strong'};
cfg.turb_cn      = {'弱湍流','强湍流'};
cfg.turb_labels  = [0, 1];

cfg.date_tag = '2026.06.28';

% 输出目录
cfg.out_root_all = fullfile(data_root, 'preprocessed_uniform_qam_rx_fast', cfg.date_tag);
if ~exist(cfg.out_root_all, 'dir'), mkdir(cfg.out_root_all); end

% 采样率
cfg.Fs_rx   = 80e9;
cfg.Fs_base = 16e9;

% 每个bin包含3帧
cfg.n_frames_per_file = 3;

% OFDM参数：必须和 tx_3frame_6mod_uniform_minimal_txt.m 保持一致
cfg.zeros_head = 80;
cfg.n_fft      = 256;
cfg.n_guard    = 16;
cfg.n_syms     = 128;
cfg.carrier_loc = 4:126;
cfg.carrier_loc_demo = [4:126, 132:254];
cfg.n_sc = length(cfg.carrier_loc);
cfg.n_sc_demo = length(cfg.carrier_loc_demo);
cfg.sym_len = cfg.n_fft + cfg.n_guard;
cfg.frame_len_16 = cfg.zeros_head + cfg.n_guard + 2*cfg.n_fft + cfg.sym_len*cfg.n_syms;
cfg.frame_pre_lts = cfg.zeros_head + cfg.n_guard - 5;
cfg.next_search_backoff = 800;

% 多表征参数
cfg.cdm_bins = 64;
cfg.cdm_clip = 3.0;
cfg.M_time = 32768;

% 快速模式开关
cfg.fast_mode = true;
cfg.save_per_frame_mat = true;
cfg.save_per_file_mat = false;
cfg.plot_quality_check = false;
cfg.max_qc_fig_per_set = 2;

% 文件筛选
cfg.max_files_per_turb_per_mod = inf;

% 并行加速：
%   true  : CPU parfor 按文件并行，推荐；
%   false : 顺序执行，可选单GPU加速FFT。
cfg.use_parallel_cpu = true;

% CUDA只在非parfor模式下建议开启
cfg.use_cuda = false;

% 测试少量文件时可改成 12；正式运行用 inf
cfg.debug_max_jobs = inf;

cfg.continue_on_frame_error = true;

%% ===================== CUDA / 并行检查 =====================
cuda_info = struct('use_cuda_requested', cfg.use_cuda, 'cuda_available', false, ...
    'device_name', '', 'note', '');

if cfg.use_parallel_cpu
    cfg.use_cuda = false;
    cuda_info.note = 'use_parallel_cpu=true, CUDA disabled to avoid GPU contention.';
end

if cfg.use_cuda
    try
        g = gpuDevice();
        cuda_info.cuda_available = true;
        cuda_info.device_name = g.Name;
        fprintf('[CUDA] GPU available: %s | TotalMemory %.2f GB\n', g.Name, g.TotalMemory/1024^3);
    catch ME
        cfg.use_cuda = false;
        cuda_info.note = ME.message;
        fprintf('[CUDA] No usable GPU. Fallback to CPU. Reason: %s\n', ME.message);
    end
else
    fprintf('[CUDA] Disabled. CPU/parfor fast mode is used.\n');
end

%% ===================== LTS =====================
LTS = make_lts_local(cfg.n_fft);

%% ===================== 发现所有任务 =====================
all_jobs = discover_all_jobs_6mod_local(cfg);

if isempty(all_jobs)
    error('No RX .bin files found. Please check rx_data paths.');
end

if isfinite(cfg.debug_max_jobs)
    all_jobs = all_jobs(1:min(numel(all_jobs), cfg.debug_max_jobs));
end

fprintf('\n============================================\n');
fprintf('Six-modulation FAST preprocessing\n');
fprintf('============================================\n');
fprintf('Date tag       : %s\n', cfg.date_tag);
fprintf('Output root    : %s\n', cfg.out_root_all);
fprintf('Total RX files : %d\n', numel(all_jobs));
fprintf('Expected frames: %d\n', numel(all_jobs) * cfg.n_frames_per_file);
fprintf('Parallel CPU   : %d\n', cfg.use_parallel_cpu);
fprintf('Save file_result.mat: %d\n', cfg.save_per_file_mat);
fprintf('Plot QC figures     : %d\n', cfg.plot_quality_check);
fprintf('============================================\n');

%% ===================== 正式处理 =====================
t_all = tic;
results = cell(numel(all_jobs), 1);

if cfg.use_parallel_cpu
    pool_ok = false;
    try
        p = gcp('nocreate');
        if isempty(p)
            parpool('local');
        end
        pool_ok = true;
    catch ME
        warning('parpool failed, fallback to sequential mode: %s', ME.message);
        pool_ok = false;
    end

    if pool_ok
        fprintf('\n[PARFOR] Processing %d RX files...\n', numel(all_jobs));
        parfor j = 1:numel(all_jobs)
            results{j} = process_one_rx_file_fast_local(all_jobs(j), cfg, LTS);
        end
    else
        cfg.use_parallel_cpu = false;
    end
end

if ~cfg.use_parallel_cpu
    fprintf('\n[FOR] Processing %d RX files sequentially...\n', numel(all_jobs));
    for j = 1:numel(all_jobs)
        fprintf('\n[%d/%d] %s | %s | %s\n', ...
            j, numel(all_jobs), all_jobs(j).mod_name, all_jobs(j).turb_name, all_jobs(j).rx_name);
        results{j} = process_one_rx_file_fast_local(all_jobs(j), cfg, LTS);

        elapsed = toc(t_all);
        avg_file = elapsed / j;
        remain = avg_file * (numel(all_jobs)-j);
        fprintf('[PROGRESS] %d/%d files | elapsed %.1f min | remaining %.1f min\n', ...
            j, numel(all_jobs), elapsed/60, remain/60);
    end
end

total_elapsed = toc(t_all);

%% ===================== 合并 manifest 和 summary =====================
manifest_file = fullfile(cfg.out_root_all, 'manifest_all.csv');
fid_manifest = fopen(manifest_file, 'w');
write_manifest_header_fast_local(fid_manifest);

global_frame_id = 0;
processed_files = 0;
failed_files = 0;
processed_frames = 0;
failed_frames = 0;
summary_rows = [];

for j = 1:numel(results)
    R = results{j};
    if isempty(R)
        failed_files = failed_files + 1;
        continue;
    end

    if isfield(R, 'file_status') && strcmp(R.file_status, 'ok')
        processed_files = processed_files + 1;
    else
        failed_files = failed_files + 1;
    end

    if isfield(R, 'n_ok_frames')
        processed_frames = processed_frames + R.n_ok_frames;
    end
    if isfield(R, 'n_failed_frames')
        failed_frames = failed_frames + R.n_failed_frames;
    end

    if isfield(R, 'summary_row')
        summary_rows = [summary_rows; R.summary_row]; %#ok<AGROW>
    end

    if isfield(R, 'frame_records')
        for k = 1:numel(R.frame_records)
            rec = R.frame_records(k);
            if strcmp(rec.status, 'ok')
                global_frame_id = global_frame_id + 1;
                rec.global_frame_id = global_frame_id;
            else
                rec.global_frame_id = NaN;
            end
            write_manifest_row_fast_local(fid_manifest, rec);
        end
    end
end

fclose(fid_manifest);

run_summary = struct();
run_summary.cfg = cfg;
run_summary.cuda_info = cuda_info;
run_summary.n_jobs = numel(all_jobs);
run_summary.processed_files = processed_files;
run_summary.failed_files = failed_files;
run_summary.processed_frames = processed_frames;
run_summary.failed_frames = failed_frames;
run_summary.actual_total_time_sec = total_elapsed;
run_summary.manifest_file = manifest_file;
run_summary.out_root_all = cfg.out_root_all;

save(fullfile(cfg.out_root_all, 'run_summary_fast.mat'), 'run_summary', 'summary_rows', '-v7.3');
write_run_summary_txt_fast_local(run_summary, fullfile(cfg.out_root_all, 'run_summary_fast.txt'));

fprintf('\n============================================\n');
fprintf(' FAST preprocessing completed\n');
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
            turb_cn     = cfg.turb_cn{ti};
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

            fprintf('[DISCOVER] %-7s | %-6s | %s | %d files\n', ...
                mod_name, turb_name, rx_dir, numel(files));

            for i = 1:numel(files)
                job_id = job_id + 1;
                all_jobs(job_id).job_id = job_id; %#ok<AGROW>
                all_jobs(job_id).mod_name = mod_name;
                all_jobs(job_id).mod_label = mod_label;
                all_jobs(job_id).turb_subdir = turb_subdir;
                all_jobs(job_id).turb_name = turb_name;
                all_jobs(job_id).turb_cn = turb_cn;
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
%% One RX file fast processing
%% =====================================================================
function file_result = process_one_rx_file_fast_local(job, cfg, LTS)

    global PROG;
    PROG.showMessagesLevel = 0;

    file_result = struct();
    file_result.job = job;
    file_result.file_status = 'failed';
    file_result.message = '';
    file_result.n_ok_frames = 0;
    file_result.n_failed_frames = 0;
    file_result.frame_records = struct([]);
    file_result.summary_row = [];

    set_out_dir = fullfile(cfg.out_root_all, job.mod_name, job.turb_name);
    if ~exist(set_out_dir, 'dir'), mkdir(set_out_dir); end

    file_base = erase(job.rx_name, '.bin');
    safe_base = regexprep(file_base, '[^\w\-]', '_');
    file_out_dir = fullfile(set_out_dir, sprintf('%s_sig%04d_%s', job.mod_name, job.sig_idx, safe_base));
    if ~exist(file_out_dir, 'dir'), mkdir(file_out_dir); end

    try
        [rx80, read_info] = read_keysight_bin_robust_real_local(job.rx_file);
        rx80 = rx80(:).';
        rx80 = rx80 - mean(rx80);
        rx80 = rx80 ./ (rms(rx80) + eps);

        rx16 = resample(rx80, cfg.Fs_base, cfg.Fs_rx);
        rx16 = rx16(:).';
        rx16 = rx16 - mean(rx16);
        rx16 = rx16 ./ (mean(abs(rx16)) + eps);

        wrap_len = min(length(rx16), 3*cfg.frame_len_16);
        rx16_ext = [rx16, rx16(1:wrap_len)];

        cursor = 1;
        frames_meta = struct([]);

        for rk = 1:cfg.n_frames_per_file
            try
                if cursor >= length(rx16_ext) - cfg.frame_len_16
                    error('cursor too close to end');
                end

                search_sig = rx16_ext(cursor:end);
                [lts_start_rel, frame_start_rel, sync_info] = find_one_frame_start_rx1_style_local(search_sig, cfg);
                lts_start_abs   = cursor + lts_start_rel - 1;
                frame_start_abs = cursor + frame_start_rel - 1;

                [rx_sc, rx_time, rx_frame16_lts, demod_info] = demod_one_frame_from_lts_start_local(rx16_ext, lts_start_abs, LTS, cfg);

                cdm64 = make_cdm_from_rxsc_local(rx_sc, cfg.cdm_bins, cfg.cdm_clip);
                Y_iq = make_iq_tensor_local(rx_sc);
                [blind_stats, blind_stats_names] = make_blind_stats_local(rx_sc);

                frame_mat = fullfile(file_out_dir, sprintf('frame_%02d.mat', rk));

                sample = struct();
                sample.rx_sc = single(rx_sc);
                sample.Y_iq = single(Y_iq);
                sample.rx_time = single(rx_time);
                sample.rx_frame16_lts = single(rx_frame16_lts);
                sample.cdm64 = single(cdm64);
                sample.blind_stats = single(blind_stats(:).');
                sample.blind_stats_names = blind_stats_names;

                sample.snr_sc_db = single(NaN(cfg.n_sc, 1));
                sample.snr_frame_db = single(NaN);
                sample.ber_sc = single(NaN(cfg.n_sc, 1));
                sample.best_tx_id = int32(-1);
                sample.txafdem_matrix = single([]);

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
                sample.sync_info = sync_info;
                sample.demod_info = demod_info;
                sample.out_mat = frame_mat;
                sample.fast_mode = true;
                sample.note = 'snr_sc_db and snr_frame_db are not computed in fast mode. Use a separate QC script if needed.';

                if cfg.save_per_frame_mat
                    save(frame_mat, '-struct', 'sample', '-v7');
                end

                file_result.n_ok_frames = file_result.n_ok_frames + 1;

                rec = make_frame_manifest_record_fast_local(job, cfg, rk, frame_mat, ...
                    'ok', '', lts_start_abs, frame_start_abs, demod_info.cfo, demod_info.n_use, blind_stats);

                file_result.frame_records = append_record_local(file_result.frame_records, rec);

                meta = struct();
                meta.rx_frame_idx = rk;
                meta.status = 'ok';
                meta.out_mat = frame_mat;
                meta.lts_start_abs = lts_start_abs;
                meta.frame_start_abs = frame_start_abs;
                meta.cfo = demod_info.cfo;
                meta.n_use = demod_info.n_use;
                frames_meta = append_record_local(frames_meta, meta);

                cursor_next = frame_start_abs + cfg.frame_len_16 - cfg.next_search_backoff;
                if cursor_next <= cursor
                    cursor_next = cursor + round(0.8 * cfg.frame_len_16);
                end
                cursor = cursor_next;

            catch ME_frame
                file_result.n_failed_frames = file_result.n_failed_frames + 1;
                rec = make_frame_manifest_record_fast_local(job, cfg, rk, '', ...
                    'failed', ME_frame.message, NaN, NaN, NaN, NaN, NaN(1,16));
                file_result.frame_records = append_record_local(file_result.frame_records, rec);

                if cfg.continue_on_frame_error
                    cursor = cursor + round(0.8 * cfg.frame_len_16);
                else
                    rethrow(ME_frame);
                end
            end
        end

        file_result.file_status = 'ok';
        file_result.read_info = read_info;
        file_result.file_out_dir = file_out_dir;
        file_result.frames_meta = frames_meta;

        if cfg.save_per_file_mat
            save(fullfile(file_out_dir, 'file_result_fast.mat'), 'file_result', '-v7.3');
        end

        if cfg.plot_quality_check
            try
                plot_file_qc_fast_local(file_out_dir, job, cfg);
            catch
            end
        end

    catch ME_file
        file_result.file_status = 'failed';
        file_result.message = ME_file.message;
        file_result.n_failed_frames = cfg.n_frames_per_file;

        rec = make_frame_manifest_record_fast_local(job, cfg, NaN, '', ...
            'failed_file', ME_file.message, NaN, NaN, NaN, NaN, NaN(1,16));
        file_result.frame_records = append_record_local(file_result.frame_records, rec);
    end

    file_result.summary_row = [ ...
        double(job.mod_label), ...
        double(job.turb_label), ...
        double(job.sig_idx), ...
        double(file_result.n_ok_frames), ...
        double(file_result.n_failed_frames) ...
    ];
end

function arr = append_record_local(arr, rec)
    if isempty(arr)
        arr = rec;
    else
        arr(end+1) = rec;
    end
end

%% =====================================================================
%% Sync / demod
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
    info.symbol_bits = symbol_bits;
end

function [rx_sc, rx_time, rx_frame16_lts, info] = demod_one_frame_from_lts_start_local(rx, lts_start, LTS, cfg)
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

    if cfg.use_cuda
        try
            fd = gather(fft(gpuArray(dn), n_fft, 1) / sqrt(n_fft));
            lts_avg = (lts1c(:) + lts2c(:)) / 2;
            lts_fd = gather(fft(gpuArray(lts_avg), n_fft) / sqrt(n_fft));
        catch
            fd = fft(dn, n_fft, 1) / sqrt(n_fft);
            lts_avg = (lts1c(:) + lts2c(:)) / 2;
            lts_fd = fft(lts_avg, n_fft) / sqrt(n_fft);
        end
    else
        fd = fft(dn, n_fft, 1) / sqrt(n_fft);
        lts_avg = (lts1c(:) + lts2c(:)) / 2;
        lts_fd = fft(lts_avg, n_fft) / sqrt(n_fft);
    end

    H = lts_fd ./ (LTS.freq(:) + 1e-12);
    H(abs(LTS.freq(:)) < 0.5) = 1;

    feq = fd ./ H;
    rx_sc = feq(cfg.carrier_loc, :);

    rx_payload_time = dp_all(:).';
    if length(rx_payload_time) >= cfg.M_time
        rx_time = rx_payload_time(1:cfg.M_time);
    else
        rx_time = [rx_payload_time, zeros(1, cfg.M_time - length(rx_payload_time))];
    end

    n_save = min(length(rx_comp), 2*n_fft + cfg.n_syms*sym_len);
    rx_frame16_lts = rx_comp(1:n_save);

    info = struct();
    info.lts_start = lts_start;
    info.cfo = cfo;
    info.n_use = n_use;
    info.nd_available = nd;
    info.data_start = data_start;
    info.remaining = remaining;
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
    p = mean(abs(z).^2);
    z = z ./ sqrt(p + eps);

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
%% Manifest helpers
%% =====================================================================
function rec = make_frame_manifest_record_fast_local(job, cfg, rx_frame_idx, out_mat, ...
    status, message, lts_start_abs, frame_start_abs, cfo, n_use, blind_stats) %#ok<INUSD>

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

function write_manifest_header_fast_local(fid)
    fprintf(fid, ['global_frame_id,status,message,mod_name,mod_label,turb_name,turb_label,turb_subdir,' ...
        'rx_file,rx_name,sig_idx,rx_frame_idx,lts_start_abs,frame_start_abs,cfo,n_use,' ...
        'amp_mean,amp_std,amp_skew,amp_kurt,papr_db,' ...
        'i_mean,i_std,i_skew,i_kurt,q_mean,q_std,q_skew,q_kurt,' ...
        'phase_diff_std,phase_concentration,iq_corr,out_mat\n']);
end

function write_manifest_row_fast_local(fid, rec)
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

function write_run_summary_txt_fast_local(run_summary, filename)
    fid = fopen(filename, 'w');
    fprintf(fid, 'Processed files : %d\n', run_summary.processed_files);
    fprintf(fid, 'Failed files    : %d\n', run_summary.failed_files);
    fprintf(fid, 'Processed frames: %d\n', run_summary.processed_frames);
    fprintf(fid, 'Failed frames   : %d\n', run_summary.failed_frames);
    fprintf(fid, 'Actual total time: %.4f sec = %.4f min = %.4f h\n', ...
        run_summary.actual_total_time_sec, run_summary.actual_total_time_sec/60, run_summary.actual_total_time_sec/3600);
    fprintf(fid, 'Manifest: %s\n', run_summary.manifest_file);
    fprintf(fid, 'Output root: %s\n', run_summary.out_root_all);
    fprintf(fid, 'Parallel CPU: %d\n', run_summary.cfg.use_parallel_cpu);
    fprintf(fid, 'CUDA requested: %d\n', run_summary.cuda_info.use_cuda_requested);
    fprintf(fid, 'CUDA available: %d\n', run_summary.cuda_info.cuda_available);
    fprintf(fid, 'CUDA device: %s\n', run_summary.cuda_info.device_name);
    fprintf(fid, 'CUDA note: %s\n', run_summary.cuda_info.note);
    fclose(fid);
end

%% =====================================================================
%% Optional QC plot
%% =====================================================================
function plot_file_qc_fast_local(file_out_dir, job, cfg) %#ok<INUSD>
    mats = dir(fullfile(file_out_dir, 'frame_*.mat'));
    if isempty(mats), return; end

    S = load(fullfile(mats(1).folder, mats(1).name), 'rx_sc', 'cdm64');
    if ~isfield(S, 'rx_sc'), return; end

    fig = figure('Visible','off','Name','QC Fast','Position',[100 100 900 420]);

    subplot(1,2,1);
    z = S.rx_sc(:);
    plot(real(z), imag(z), 'b.', 'MarkerSize', 3);
    axis square; grid on;
    title(sprintf('%s %s RX constellation', job.mod_name, job.turb_name));
    xlabel('I'); ylabel('Q');

    subplot(1,2,2);
    if isfield(S, 'cdm64')
        imagesc(S.cdm64); axis image; colorbar;
        title('CDM 64x64');
    end

    out_png = fullfile(file_out_dir, 'fig_qc_fast.png');
    try
        exportgraphics(fig, out_png, 'Resolution', 150);
    catch
        try
            print(fig, out_png, '-dpng', '-opengl');
        catch
        end
    end
    close(fig);
end

%% =====================================================================
%% LTS / sorting helpers
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
