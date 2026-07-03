%% batch_preprocess_uniform_16qam_rx_cuda.m
% =========================================================
% Uniform QAM RX 批量预处理脚本：16QAM weak/strong turbulence
%
% 输入：
%   RX弱湍流：...\rx_data\2026.06.28\16QAM\sub01
%   RX强湍流：...\rx_data\2026.06.28\16QAM\sub03
%   TX参考：tx_3frame_6mod_uniform_txt\16QAM\sub1/sub2/...\sig_xxxx_frame1/2/3.mat
%
% 输出：
%   1. 每个解调帧一个 .mat，便于后续 Python/MATLAB 识别模型读取
%   2. manifest.csv：记录每个样本对应的调制/湍流/SNR/路径/状态
%   3. run_summary.mat / csv：整体统计
%   4. 可选少量质检图：星座图、子载波SNR图
%
% 说明：
%   - 保持与 test_one_rx_uniform_qam_3frames_verify.m 一致的同步/解调/SNR计算逻辑
%   - 默认每个 RX .bin 提取 3 帧
%   - 默认使用 CUDA(gpuArray) 加速 FFT 部分；若无 GPU 自动回退 CPU
%   - I/O、同步、symDemapper/EVM_eval 仍在 CPU 上执行，CUDA只加速局部FFT，不保证全流程线性加速
% =========================================================

clear; clear global; close all; clc;

%% ===================== 基础路径 =====================
project_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master';
data_root    = fullfile(project_root, '2_Data_Results');
addpath(genpath(project_root));

global PROG;
PROG.showMessagesLevel = 1;
try
    initProg();
catch
    warning('initProg() failed or not found. Continue anyway.');
end

%% ===================== 用户配置 =====================
cfg = struct();
cfg.project_root = project_root;
cfg.data_root    = data_root;

cfg.mod_name = '64QAM';
cfg.mod_label = 1;  % zero-based label: QPSK=0,16QAM=1,32QAM=2,64QAM=3,128QAM=4,256QAM=5

% RX目录：sub01弱湍流，sub03强湍流
cfg.rx_sets = struct( ...
    'rx_dir',   { ...
        fullfile(data_root, 'rx_data', '2026.06.28', '64QAM', 'sub01'), ...
        fullfile(data_root, 'rx_data', '2026.06.28', '64QAM', 'sub03')}, ...
    'turb_name', {'weak', 'strong'}, ...
    'turb_cn',   {'弱湍流', '强湍流'}, ...
    'turb_label',{0, 1}, ...
    'tx_preferred_sub', {'sub1', 'sub1'} ...
    );

% TX参考根目录：脚本会在 sub1/sub2/... 中自动找 sig_xxxx_frame1/2/3.mat
cfg.tx_root_mod = fullfile(data_root, 'tx_3frame_6mod_uniform_txt', cfg.mod_name);

% 输出目录
cfg.out_root = fullfile(data_root, 'preprocessed_uniform_qam_rx', '2026.06.28', cfg.mod_name);
if ~exist(cfg.out_root, 'dir'), mkdir(cfg.out_root); end

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
cfg.shift_set = -5:5;
cfg.try_conjugate = false;

% CDM图像参数，用于后续识别可选输入
cfg.cdm_bins = 64;
cfg.cdm_clip = 3.0;
cfg.M_time = 32768;

% CUDA / 并行 / 画图
cfg.use_cuda = true;          % 若有GPU则用gpuArray加速FFT；无GPU自动回退
cfg.use_parallel_cpu = false; % 当前建议false；如果打开，自动关闭CUDA，避免多个worker抢同一块GPU
cfg.plot_quality_check = true;
cfg.max_qc_fig_per_set = 3;   % 每个湍流目录最多画几个文件的图，避免批处理太慢
cfg.save_per_file_mat = true;
cfg.save_per_frame_mat = true;

% 运行时长估计：先处理每个湍流目录前N个文件估计速度
cfg.benchmark_n_files_per_set = 3;

% 文件筛选；为空表示处理目录下所有 .bin
cfg.max_files_per_set = inf;

%% ===================== CUDA 检查 =====================
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
    fprintf('[CUDA] Disabled by cfg.use_cuda=false or parallel CPU mode.\n');
end

%% ===================== OFDM / QAM / LTS =====================
ofdm = struct();
ofdm.NumberOfIFFTSamples = cfg.n_fft;
ofdm.NumberOfGuardTime = cfg.n_guard;
ofdm.Carrier_location = cfg.carrier_loc;
ofdm.Carrier_location_demo = cfg.carrier_loc_demo;
ofdm.NumberOfCarriers = cfg.n_sc;
ofdm.NumberOfCarriers_demo = cfg.n_sc_demo;
ofdm.size = cfg.n_syms;

[Mq, bits] = mod_to_order_bits_local(cfg.mod_name);
nBpS_net = bits;  % Uniform QAM，不使用CCDM shaping overhead
TX.SIG = setSignalParams('symRate', 8e9, 'M', Mq, ...
    'nPol', 1, 'nBpS', nBpS_net, 'nSyms', cfg.n_syms, ...
    'roll-off', 0.25, 'modulation', 'QAM');
TX.QAM = QAM_config(TX.SIG);
C = TX.QAM.IQmap;

DSP = struct();
DSP.DEMAPPER.normMethod = 'MMSE';

LTS = make_lts_local(cfg.n_fft);

%% ===================== 文件发现 =====================
all_jobs = struct([]);
job_id = 0;

for s = 1:numel(cfg.rx_sets)
    rx_dir = cfg.rx_sets(s).rx_dir;
    if ~exist(rx_dir, 'dir')
        warning('RX dir not found: %s', rx_dir);
        continue;
    end

    files = dir(fullfile(rx_dir, '*.bin'));
    files = sort_files_by_number_local(files);
    if isfinite(cfg.max_files_per_set)
        files = files(1:min(numel(files), cfg.max_files_per_set));
    end

    fprintf('\n[DISCOVER] %s (%s): %d RX files\n', cfg.rx_sets(s).turb_name, rx_dir, numel(files));

    for i = 1:numel(files)
        job_id = job_id + 1;
        all_jobs(job_id).job_id = job_id; %#ok<SAGROW>
        all_jobs(job_id).set_idx = s;
        all_jobs(job_id).rx_file = fullfile(files(i).folder, files(i).name);
        all_jobs(job_id).rx_name = files(i).name;
        all_jobs(job_id).sig_idx = infer_sig_idx_from_filename_local(files(i).name, i);
        all_jobs(job_id).turb_name = cfg.rx_sets(s).turb_name;
        all_jobs(job_id).turb_cn = cfg.rx_sets(s).turb_cn;
        all_jobs(job_id).turb_label = cfg.rx_sets(s).turb_label;
        all_jobs(job_id).tx_preferred_sub = cfg.rx_sets(s).tx_preferred_sub;
    end
end

if isempty(all_jobs)
    error('No RX .bin files found. Please check cfg.rx_sets paths.');
end

fprintf('\nTotal jobs: %d RX files. Expected frames up to %d.\n', ...
    numel(all_jobs), numel(all_jobs)*cfg.n_frames_per_file);

%% ===================== 运行时长估计 =====================
bench_jobs = select_benchmark_jobs_local(all_jobs, cfg);

fprintf('\n============================================\n');
fprintf(' Benchmark for runtime estimation\n');
fprintf('============================================\n');
fprintf('Benchmark files: %d\n', numel(bench_jobs));

t_bench = tic;
bench_status = cell(numel(bench_jobs),1);
for b = 1:numel(bench_jobs)
    try
        tmp_out = fullfile(cfg.out_root, '_benchmark_tmp');
        if ~exist(tmp_out, 'dir'), mkdir(tmp_out); end
        process_one_rx_file_local(bench_jobs(b), cfg, ofdm, LTS, C, DSP, tmp_out, false);
        bench_status{b} = 'ok';
    catch ME
        bench_status{b} = ['failed: ' ME.message];
    end
end
bench_time = toc(t_bench);
bench_per_file = bench_time / max(numel(bench_jobs), 1);
est_total_time = bench_per_file * numel(all_jobs);

fprintf('Benchmark elapsed: %.2f s for %d files\n', bench_time, numel(bench_jobs));
fprintf('Estimated avg time/file: %.2f s\n', bench_per_file);
fprintf('Estimated total time: %.1f s = %.1f min = %.2f h\n', ...
    est_total_time, est_total_time/60, est_total_time/3600);

% 删除benchmark临时目录中的大文件，避免混淆
try
    rmdir(fullfile(cfg.out_root, '_benchmark_tmp'), 's');
catch
end

%% ===================== 正式批处理 =====================
fprintf('\n============================================\n');
fprintf(' Batch preprocessing started\n');
fprintf('============================================\n');

t_all = tic;

manifest_file = fullfile(cfg.out_root, 'manifest.csv');
fid_manifest = fopen(manifest_file, 'w');
write_manifest_header_local(fid_manifest);

summary_rows = [];
frame_counter = 0;
processed_files = 0;
failed_files = 0;
processed_frames = 0;
failed_frames = 0;

for j = 1:numel(all_jobs)
    job = all_jobs(j);

    fprintf('\n[%d/%d] %s | %s | sig_idx=%d\n', ...
        j, numel(all_jobs), job.turb_name, job.rx_name, job.sig_idx);

    set_out_dir = fullfile(cfg.out_root, job.turb_name);
    if ~exist(set_out_dir, 'dir'), mkdir(set_out_dir); end

    do_plot = cfg.plot_quality_check && count_qc_files_done_local(set_out_dir) < cfg.max_qc_fig_per_set;

    try
        [file_result, frame_records] = process_one_rx_file_local(job, cfg, ofdm, LTS, C, DSP, set_out_dir, do_plot);
        processed_files = processed_files + 1;

        for r = 1:numel(frame_records)
            frame_counter = frame_counter + 1;
            frame_records(r).global_frame_id = frame_counter;
            write_manifest_row_local(fid_manifest, frame_records(r));
        end

        processed_frames = processed_frames + file_result.n_ok_frames;
        failed_frames = failed_frames + file_result.n_failed_frames;
        summary_rows = [summary_rows; file_result_to_summary_row_local(file_result)]; %#ok<AGROW>

        fprintf('  OK frames: %d/%d | file_snr=%.2f dB\n', ...
            file_result.n_ok_frames, cfg.n_frames_per_file, file_result.file_snr_db);

    catch ME
        failed_files = failed_files + 1;
        failed_frames = failed_frames + cfg.n_frames_per_file;

        rec = make_failed_file_manifest_record_local(job, cfg, ME.message);
        write_manifest_row_local(fid_manifest, rec);

        fprintf('  FAILED: %s\n', ME.message);
    end

    if mod(j, 5) == 0 || j == numel(all_jobs)
        elapsed = toc(t_all);
        avg_file = elapsed / j;
        remain = avg_file * (numel(all_jobs)-j);
        fprintf('\n[PROGRESS] %d/%d files | elapsed %.1f min | remaining %.1f min\n', ...
            j, numel(all_jobs), elapsed/60, remain/60);
    end
end

fclose(fid_manifest);

total_elapsed = toc(t_all);

%% ===================== 保存汇总 =====================
run_summary = struct();
run_summary.cfg = cfg;
run_summary.cuda_info = cuda_info;
run_summary.n_jobs = numel(all_jobs);
run_summary.processed_files = processed_files;
run_summary.failed_files = failed_files;
run_summary.processed_frames = processed_frames;
run_summary.failed_frames = failed_frames;
run_summary.benchmark_time_sec = bench_time;
run_summary.benchmark_per_file_sec = bench_per_file;
run_summary.estimated_total_time_sec = est_total_time;
run_summary.actual_total_time_sec = total_elapsed;
run_summary.manifest_file = manifest_file;

save(fullfile(cfg.out_root, 'run_summary.mat'), 'run_summary', 'summary_rows', '-v7.3');
write_run_summary_txt_local(run_summary, fullfile(cfg.out_root, 'run_summary.txt'));

fprintf('\n============================================\n');
fprintf(' Batch preprocessing completed\n');
fprintf('============================================\n');
fprintf('Processed files : %d\n', processed_files);
fprintf('Failed files    : %d\n', failed_files);
fprintf('Processed frames: %d\n', processed_frames);
fprintf('Failed frames   : %d\n', failed_frames);
fprintf('Estimated time  : %.1f min\n', est_total_time/60);
fprintf('Actual time     : %.1f min\n', total_elapsed/60);
fprintf('Manifest        : %s\n', manifest_file);
fprintf('Out root        : %s\n', cfg.out_root);
fprintf('============================================\n');

%% =====================================================================
%% Main processing function
%% =====================================================================
function [file_result, frame_records] = process_one_rx_file_local(job, cfg, ofdm, LTS, C, DSP, set_out_dir, do_plot)

    frame_records = struct([]);

    [tx_refs, tx_ref_files, tx_dir_used] = load_uniform_tx_frame_refs_batch_local(cfg, job.sig_idx, job.tx_preferred_sub);

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
    frames = struct([]);
    n_ok = 0;
    n_failed = 0;

    file_base = erase(job.rx_name, '.bin');
    file_out_dir = fullfile(set_out_dir, sprintf('%s_sig%04d_%s', cfg.mod_name, job.sig_idx, file_base));
    if ~exist(file_out_dir, 'dir'), mkdir(file_out_dir); end

    for rk = 1:cfg.n_frames_per_file
        try
            if cursor >= length(rx16_ext) - cfg.frame_len_16
                error('cursor too close to end');
            end

            search_sig = rx16_ext(cursor:end);
            [lts_start_rel, frame_start_rel, sync_info] = find_one_frame_start_rx1_style_local(search_sig, ofdm, cfg);
            lts_start_abs   = cursor + lts_start_rel - 1;
            frame_start_abs = cursor + frame_start_rel - 1;

            [rx_sc, rx_time, rx_frame16_lts, demod_info] = demod_one_frame_from_lts_start_local(rx16_ext, lts_start_abs, LTS, cfg);

            [best_tx_id, best_snr_db, best_snr_sc_db, snr_list_db, best_txafdem, best_ber_sc, align_info] = ...
                match_rx_to_tx_by_rx1_snr_local(rx_sc, tx_refs, C, DSP, cfg);

            cdm64 = make_cdm_from_rxsc_local(rx_sc, cfg.cdm_bins, cfg.cdm_clip);

            n_ok = n_ok + 1;
            one = struct();
            one.status = 'ok';
            one.message = '';
            one.rx_frame_idx = rk;
            one.rx_sc = single(rx_sc);
            one.Y_iq = make_iq_tensor_local(rx_sc);
            one.rx_time = single(rx_time);
            one.rx_frame16_lts = single(rx_frame16_lts);
            one.cdm64 = single(cdm64);
            one.best_tx_id = best_tx_id;
            one.snr_frame_rx1_db = best_snr_db;
            one.snr_sc_rx1_db = single(best_snr_sc_db);
            one.snr_list_db = single(snr_list_db);
            one.txafdem_matrix = single(best_txafdem);
            one.ber_sc = single(best_ber_sc);
            one.sync_info = sync_info;
            one.demod_info = demod_info;
            one.align_info = align_info;
            one.lts_start_abs = lts_start_abs;
            one.frame_start_abs = frame_start_abs;

            frames = [frames, one]; %#ok<AGROW>

            frame_mat = fullfile(file_out_dir, sprintf('frame_%02d.mat', rk));
            if cfg.save_per_frame_mat
                sample = make_recognition_sample_local(job, cfg, one, tx_ref_files, tx_dir_used, frame_mat);
                save(frame_mat, '-struct', 'sample', '-v7');
            end

            rec = make_frame_manifest_record_local(job, cfg, one, frame_mat, tx_dir_used, tx_ref_files);
            frame_records = [frame_records, rec]; %#ok<AGROW>

            cursor_next = frame_start_abs + cfg.frame_len_16 - cfg.next_search_backoff;
            if cursor_next <= cursor
                cursor_next = cursor + round(0.8 * cfg.frame_len_16);
            end
            cursor = cursor_next;

        catch ME
            n_failed = n_failed + 1;
            one = struct();
            one.status = 'failed';
            one.message = ME.message;
            one.rx_frame_idx = rk;
            frames = [frames, one]; %#ok<AGROW>

            rec = make_failed_frame_manifest_record_local(job, cfg, rk, ME.message);
            frame_records = [frame_records, rec]; %#ok<AGROW>

            cursor = cursor + round(0.8 * cfg.frame_len_16);
        end
    end

    frame_snr = NaN(1, numel(frames));
    best_order = NaN(1, numel(frames));
    for k = 1:numel(frames)
        if isfield(frames(k), 'snr_frame_rx1_db')
            frame_snr(k) = frames(k).snr_frame_rx1_db;
        end
        if isfield(frames(k), 'best_tx_id')
            best_order(k) = frames(k).best_tx_id;
        end
    end

    valid_snr = frame_snr(isfinite(frame_snr));
    if isempty(valid_snr)
        file_snr_db = NaN;
    else
        file_snr_db = 10*log10(mean(10.^(valid_snr/10)));
    end

    file_result = struct();
    file_result.job = job;
    file_result.tx_dir_used = tx_dir_used;
    file_result.tx_ref_files = tx_ref_files;
    file_result.read_info = read_info;
    file_result.n_ok_frames = n_ok;
    file_result.n_failed_frames = n_failed;
    file_result.frame_snr = frame_snr;
    file_result.best_order = best_order;
    file_result.file_snr_db = file_snr_db;
    file_result.frames = frames;
    file_result.file_out_dir = file_out_dir;

    if cfg.save_per_file_mat
        save(fullfile(file_out_dir, 'file_result.mat'), 'file_result', '-v7.3');
    end

    write_frame_snr_csv_batch_local(file_result, fullfile(file_out_dir, 'frame_snr_summary.csv'));
    write_subcarrier_snr_csv_batch_local(file_result, fullfile(file_out_dir, 'subcarrier_snr_by_frame.csv'));

    if do_plot
        plot_file_qc_local(file_result, cfg, file_out_dir);
    end
end

%% =====================================================================
%% TX reference / sample saving
%% =====================================================================
function [tx_refs, tx_ref_files, tx_dir_used] = load_uniform_tx_frame_refs_batch_local(cfg, sig_idx, preferred_sub)

    candidates = {};
    if ~isempty(preferred_sub)
        candidates{end+1} = fullfile(cfg.tx_root_mod, preferred_sub); %#ok<AGROW>
    end

    d = dir(cfg.tx_root_mod);
    for i = 1:numel(d)
        if d(i).isdir && ~startsWith(d(i).name, '.')
            p = fullfile(d(i).folder, d(i).name);
            if ~any(strcmp(candidates, p))
                candidates{end+1} = p; %#ok<AGROW>
            end
        end
    end

    tx_refs = cell(1, cfg.n_frames_per_file);
    tx_ref_files = cell(1, cfg.n_frames_per_file);
    tx_dir_used = '';

    for c = 1:numel(candidates)
        ok = true;
        tmp_refs = cell(1, cfg.n_frames_per_file);
        tmp_files = cell(1, cfg.n_frames_per_file);
        for tid = 1:cfg.n_frames_per_file
            f = fullfile(candidates{c}, sprintf('sig_%04d_frame%d.mat', sig_idx, tid));
            tmp_files{tid} = f;
            if ~exist(f, 'file')
                ok = false;
                break;
            end
            tmp = load(f);
            if ~isfield(tmp, 'data_tx')
                ok = false;
                break;
            end
            x = tmp.data_tx;
            if size(x,1) == cfg.n_syms && size(x,2) == cfg.n_sc
                tmp_refs{tid} = x.';
            elseif size(x,1) == cfg.n_sc && size(x,2) == cfg.n_syms
                tmp_refs{tid} = x;
            else
                x = x(:);
                if numel(x) < cfg.n_sc * cfg.n_syms
                    ok = false;
                    break;
                end
                tmp_refs{tid} = reshape(x(1:cfg.n_sc*cfg.n_syms), cfg.n_sc, cfg.n_syms);
            end
        end
        if ok
            tx_refs = tmp_refs;
            tx_ref_files = tmp_files;
            tx_dir_used = candidates{c};
            return;
        end
    end

    error('TX frame references not found for sig_%04d under %s', sig_idx, cfg.tx_root_mod);
end

function sample = make_recognition_sample_local(job, cfg, one, tx_ref_files, tx_dir_used, frame_mat)
    sample = struct();
    sample.Y_iq = one.Y_iq;                         % [2, n_sc, n_sym], single
    sample.rx_sc = one.rx_sc;                       % complex [n_sc, n_sym], single
    sample.cdm64 = one.cdm64;                       % [64,64], single
    sample.snr_sc_db = one.snr_sc_rx1_db;           % [n_sc,1], single
    sample.snr_frame_db = single(one.snr_frame_rx1_db);
    sample.ber_sc = one.ber_sc;
    sample.mod_name = cfg.mod_name;
    sample.mod_label = int32(cfg.mod_label);
    sample.turb_name = job.turb_name;
    sample.turb_label = int32(job.turb_label);
    sample.rx_file = job.rx_file;
    sample.rx_name = job.rx_name;
    sample.sig_idx = int32(job.sig_idx);
    sample.rx_frame_idx = int32(one.rx_frame_idx);
    sample.best_tx_id = int32(one.best_tx_id);
    sample.tx_dir_used = tx_dir_used;
    sample.tx_ref_files = tx_ref_files;
    sample.align_shift = int32(one.align_info.shift);
    sample.rx_variant = one.align_info.rx_variant;
    sample.lts_start_abs = int64(one.lts_start_abs);
    sample.frame_start_abs = int64(one.frame_start_abs);
    sample.out_mat = frame_mat;
end

function Y_iq = make_iq_tensor_local(rx_sc)
    Y_iq = zeros(2, size(rx_sc,1), size(rx_sc,2), 'single');
    Y_iq(1,:,:) = single(real(rx_sc));
    Y_iq(2,:,:) = single(imag(rx_sc));
end

%% =====================================================================
%% Manifest helpers
%% =====================================================================
function write_manifest_header_local(fid)
    fprintf(fid, ['global_frame_id,status,message,mod_name,mod_label,turb_name,turb_label,' ...
        'rx_file,rx_name,sig_idx,rx_frame_idx,best_tx_id,snr_frame_db,' ...
        'snr_sc_mean_db,snr_sc_median_db,n_valid_sc,align_shift,rx_variant,' ...
        'tx_dir_used,out_mat\n']);
end

function rec = make_frame_manifest_record_local(job, cfg, one, frame_mat, tx_dir_used, tx_ref_files) %#ok<INUSD>
    s = double(one.snr_sc_rx1_db(:));
    valid = s(isfinite(s));
    rec = struct();
    rec.global_frame_id = NaN;
    rec.status = 'ok';
    rec.message = '';
    rec.mod_name = cfg.mod_name;
    rec.mod_label = cfg.mod_label;
    rec.turb_name = job.turb_name;
    rec.turb_label = job.turb_label;
    rec.rx_file = job.rx_file;
    rec.rx_name = job.rx_name;
    rec.sig_idx = job.sig_idx;
    rec.rx_frame_idx = one.rx_frame_idx;
    rec.best_tx_id = one.best_tx_id;
    rec.snr_frame_db = one.snr_frame_rx1_db;
    if isempty(valid)
        rec.snr_sc_mean_db = NaN;
        rec.snr_sc_median_db = NaN;
        rec.n_valid_sc = 0;
    else
        rec.snr_sc_mean_db = 10*log10(mean(10.^(valid/10)));
        rec.snr_sc_median_db = median(valid);
        rec.n_valid_sc = numel(valid);
    end
    rec.align_shift = one.align_info.shift;
    rec.rx_variant = one.align_info.rx_variant;
    rec.tx_dir_used = tx_dir_used;
    rec.out_mat = frame_mat;
end

function rec = make_failed_frame_manifest_record_local(job, cfg, rk, msg)
    rec = struct();
    rec.global_frame_id = NaN;
    rec.status = 'failed';
    rec.message = sanitize_csv_text_local(msg);
    rec.mod_name = cfg.mod_name;
    rec.mod_label = cfg.mod_label;
    rec.turb_name = job.turb_name;
    rec.turb_label = job.turb_label;
    rec.rx_file = job.rx_file;
    rec.rx_name = job.rx_name;
    rec.sig_idx = job.sig_idx;
    rec.rx_frame_idx = rk;
    rec.best_tx_id = NaN;
    rec.snr_frame_db = NaN;
    rec.snr_sc_mean_db = NaN;
    rec.snr_sc_median_db = NaN;
    rec.n_valid_sc = 0;
    rec.align_shift = NaN;
    rec.rx_variant = 'none';
    rec.tx_dir_used = '';
    rec.out_mat = '';
end

function rec = make_failed_file_manifest_record_local(job, cfg, msg)
    rec = make_failed_frame_manifest_record_local(job, cfg, NaN, msg);
end

function write_manifest_row_local(fid, rec)
    fprintf(fid, '%g,%s,%s,%s,%d,%s,%d,%s,%s,%d,%g,%g,%.12g,%.12g,%.12g,%d,%g,%s,%s,%s\n', ...
        rec.global_frame_id, rec.status, sanitize_csv_text_local(rec.message), ...
        rec.mod_name, rec.mod_label, rec.turb_name, rec.turb_label, ...
        sanitize_csv_text_local(rec.rx_file), sanitize_csv_text_local(rec.rx_name), rec.sig_idx, ...
        rec.rx_frame_idx, rec.best_tx_id, rec.snr_frame_db, rec.snr_sc_mean_db, rec.snr_sc_median_db, ...
        rec.n_valid_sc, rec.align_shift, rec.rx_variant, ...
        sanitize_csv_text_local(rec.tx_dir_used), sanitize_csv_text_local(rec.out_mat));
end

function t = sanitize_csv_text_local(t)
    if isempty(t), t = ''; return; end
    if isstring(t), t = char(t); end
    t = strrep(t, ',', ';');
    t = strrep(t, newline, ' ');
    t = strrep(t, sprintf('\r'), ' ');
    t = strrep(t, sprintf('\n'), ' ');
end

function row = file_result_to_summary_row_local(file_result)
    row = [file_result.job.set_idx, file_result.job.sig_idx, file_result.job.turb_label, ...
        file_result.n_ok_frames, file_result.n_failed_frames, file_result.file_snr_db];
end

function write_run_summary_txt_local(run_summary, filename)
    fid = fopen(filename, 'w');
    fprintf(fid, 'Processed files : %d\n', run_summary.processed_files);
    fprintf(fid, 'Failed files    : %d\n', run_summary.failed_files);
    fprintf(fid, 'Processed frames: %d\n', run_summary.processed_frames);
    fprintf(fid, 'Failed frames   : %d\n', run_summary.failed_frames);
    fprintf(fid, 'Benchmark avg time/file: %.4f sec\n', run_summary.benchmark_per_file_sec);
    fprintf(fid, 'Estimated total time: %.4f sec = %.4f min\n', run_summary.estimated_total_time_sec, run_summary.estimated_total_time_sec/60);
    fprintf(fid, 'Actual total time: %.4f sec = %.4f min\n', run_summary.actual_total_time_sec, run_summary.actual_total_time_sec/60);
    fprintf(fid, 'Manifest: %s\n', run_summary.manifest_file);
    fprintf(fid, 'CUDA requested: %d\n', run_summary.cuda_info.use_cuda_requested);
    fprintf(fid, 'CUDA available: %d\n', run_summary.cuda_info.cuda_available);
    fprintf(fid, 'CUDA device: %s\n', run_summary.cuda_info.device_name);
    fprintf(fid, 'CUDA note: %s\n', run_summary.cuda_info.note);
    fclose(fid);
end

function write_frame_snr_csv_batch_local(file_result, filename)
    fid = fopen(filename, 'w');
    fprintf(fid, 'rx_frame,best_tx_frame,snr_db\n');
    for i = 1:numel(file_result.frame_snr)
        fprintf(fid, '%d,%g,%.12g\n', i, file_result.best_order(i), file_result.frame_snr(i));
    end
    fclose(fid);
end

function write_subcarrier_snr_csv_batch_local(file_result, filename)
    fid = fopen(filename, 'w');
    fprintf(fid, 'rx_frame,subcarrier_idx,snr_sc_db\n');
    for f = 1:numel(file_result.frames)
        if isfield(file_result.frames(f), 'snr_sc_rx1_db')
            s = file_result.frames(f).snr_sc_rx1_db;
            for k = 1:numel(s)
                fprintf(fid, '%d,%d,%.12g\n', f, k, s(k));
            end
        end
    end
    fclose(fid);
end

%% =====================================================================
%% Plot helpers
%% =====================================================================
function n = count_qc_files_done_local(set_out_dir)
    d = dir(fullfile(set_out_dir, '**', 'fig_qc_summary.png'));
    n = numel(d);
end

function plot_file_qc_local(file_result, cfg, out_dir)
    ok = arrayfun(@(x)isfield(x,'rx_sc'), file_result.frames);
    if ~any(ok), return; end

    all_rx = [];
    all_dec = [];
    snr_mat = NaN(cfg.n_sc, numel(file_result.frames));

    for f = 1:numel(file_result.frames)
        if isfield(file_result.frames(f), 'rx_sc')
            all_rx = [all_rx; file_result.frames(f).rx_sc(:)]; %#ok<AGROW>
        end
        if isfield(file_result.frames(f), 'txafdem_matrix') && ~isempty(file_result.frames(f).txafdem_matrix)
            all_dec = [all_dec; file_result.frames(f).txafdem_matrix(:)]; %#ok<AGROW>
        end
        if isfield(file_result.frames(f), 'snr_sc_rx1_db')
            s = file_result.frames(f).snr_sc_rx1_db;
            snr_mat(1:numel(s), f) = double(s(:));
        end
    end

    fig = figure('Visible','off','Name','QC Summary','Position',[100 100 1400 760]);
    subplot(2,2,1);
    plot(all_rx, 'b.', 'MarkerSize', 3); axis square; grid on;
    title(sprintf('%s %s RX eq constellation', cfg.mod_name, file_result.job.turb_name)); xlabel('I'); ylabel('Q');

    subplot(2,2,2);
    if ~isempty(all_dec)
        plot(all_dec, 'r.', 'MarkerSize', 3); axis square; grid on;
        title('Decision / txafdem'); xlabel('I'); ylabel('Q');
    else
        text(0.1,0.5,'No decision constellation'); axis off;
    end

    subplot(2,2,3); hold on;
    for f = 1:size(snr_mat,2)
        plot(1:cfg.n_sc, snr_mat(:,f), '-', 'LineWidth', 1.0);
    end
    snr_avg = 10*log10(mean(10.^(snr_mat/10), 2, 'omitnan'));
    plot(1:cfg.n_sc, snr_avg, 'k-', 'LineWidth', 2.0);
    grid on; xlabel('Subcarrier index'); ylabel('SNR per subcarrier (dB)');
    title(sprintf('Subcarrier SNR | file SNR %.2f dB', file_result.file_snr_db));

    subplot(2,2,4);
    bar(file_result.frame_snr); grid on;
    xlabel('Frame'); ylabel('Frame SNR (dB)'); title('Frame SNR');

    out_png = fullfile(out_dir, 'fig_qc_summary.png');
    try
        exportgraphics(fig, out_png, 'Resolution', 150);
    catch
        try
            print(fig, out_png, '-dpng', '-opengl');
        catch ME
            warning('Failed to save QC figure: %s', ME.message);
        end
    end
    close(fig);
end

%% =====================================================================
%% Sorting / benchmark helpers
%% =====================================================================
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

function bench_jobs = select_benchmark_jobs_local(all_jobs, cfg)
    keep = false(1, numel(all_jobs));
    set_ids = unique([all_jobs.set_idx]);
    for s = set_ids
        idx = find([all_jobs.set_idx] == s);
        idx = idx(1:min(numel(idx), cfg.benchmark_n_files_per_set));
        keep(idx) = true;
    end
    bench_jobs = all_jobs(keep);
end

%% =====================================================================
%% Mod/QAM/LTS helpers
%% =====================================================================
function [Mq, bits] = mod_to_order_bits_local(mod_name)
    switch mod_name
        case 'QPSK'
            Mq = 4; bits = 2;
        case '16QAM'
            Mq = 16; bits = 4;
        case '32QAM'
            Mq = 32; bits = 5;
        case '64QAM'
            Mq = 64; bits = 6;
        case '128QAM'
            Mq = 128; bits = 7;
        case '256QAM'
            Mq = 256; bits = 8;
        otherwise
            error('Unknown modulation: %s', mod_name);
    end
end

function LTS = make_lts_local(n_fft)
    load('LongTrainSym_ini.mat', 'LongTrainSym_ini');
    LTS_f0 = LongTrainSym_ini(1:n_fft);
    LTS_f0([1 n_fft/2+1]) = 0;
    ltrs_in = LTS_f0;
    ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));
    LTS.freq = ltrs_in(:);
    LTS.time = ifft(ltrs_in(:));
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

    error('read_keysight_bin_robust_real failed: standard=[%s], infer=[%s]', info.standard_error, info.infer_error);
end

function [y, info] = read_keysight_real_standard_or_infer_local(filename, force_infer)
    fid = fopen(filename, 'rb', 'ieee-le');
    if fid == -1, error('Cannot open: %s', filename); end
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
            if length(raw) < 1000, error('raw too short'); end
            if std(raw(1:min(5000,end))) == 0, error('zero variance raw'); end
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

%% =====================================================================
%% Sync / demod
%% =====================================================================
function [lts_start, frame_start, info] = find_one_frame_start_rx1_style_local(rx, ofdm, cfg)
    rx = rx(:).';
    n_fft = ofdm.NumberOfIFFTSamples;
    n_guard = ofdm.NumberOfGuardTime;
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
    [fine_time_est, data_df, max_peak_long] = rx_fine_time_sync_cross_corr(detected_packet, n_guard, ltrs_in, cfg.zeros_head, 0);
    lts_start = edge_index + fine_time_est - 1;
    frame_start = lts_start - cfg.frame_pre_lts;
    if frame_start < 1, frame_start = 1; end
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
    if nd <= 0, error('no complete OFDM symbols: nd=0'); end
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
%% SNR matching / EVM
%% =====================================================================
function [best_tx_id, best_snr_db, best_snr_sc_db, snr_list_db, best_txafdem, best_ber_sc, align_info] = ...
    match_rx_to_tx_by_rx1_snr_local(rx_sc, tx_refs, C, DSP_template, cfg)

    n_ref = length(tx_refs);
    snr_list_db = NaN(1, n_ref);
    best_global_snr = -inf;
    best_tx_id = NaN;
    best_snr_sc_db = NaN(size(rx_sc,1),1);
    best_txafdem = [];
    best_ber_sc = NaN(size(rx_sc,1),1);
    align_info = struct('shift', NaN, 'rx_variant', 'none');

    for tid = 1:n_ref
        tx_ref0 = normalize_tx_ref_shape_for_rx1_local(tx_refs{tid}, size(rx_sc,1));
        if cfg.try_conjugate
            rx_variants = {rx_sc, conj(rx_sc)};
            rx_names = {'normal','conj'};
        else
            rx_variants = {rx_sc};
            rx_names = {'normal'};
        end

        best_tid_snr = -inf;
        best_tid_sc = NaN(size(rx_sc,1),1);
        best_tid_txafdem = [];
        best_tid_ber = NaN(size(rx_sc,1),1);
        best_tid_info = struct('shift', NaN, 'rx_variant', 'none');

        for rv = 1:length(rx_variants)
            rx0 = rx_variants{rv};
            for sh = cfg.shift_set
                [rx_use, tx_use] = align_rx_tx_by_symbol_shift_local(rx0, tx_ref0, sh);
                if isempty(rx_use), continue; end
                try
                    [snr_frame_db, snr_sc_db, txafdem_matrix, ber_sc] = compute_rx1_style_snr_local(rx_use, tx_use, C, DSP_template);
                    if isfinite(snr_frame_db) && snr_frame_db > best_tid_snr
                        best_tid_snr = snr_frame_db;
                        best_tid_sc = snr_sc_db;
                        best_tid_txafdem = txafdem_matrix;
                        best_tid_ber = ber_sc;
                        best_tid_info.shift = sh;
                        best_tid_info.rx_variant = rx_names{rv};
                    end
                catch
                    continue;
                end
            end
        end

        snr_list_db(tid) = best_tid_snr;
        if isfinite(best_tid_snr) && best_tid_snr > best_global_snr
            best_global_snr = best_tid_snr;
            best_tx_id = tid;
            best_snr_sc_db = best_tid_sc;
            best_txafdem = best_tid_txafdem;
            best_ber_sc = best_tid_ber;
            align_info = best_tid_info;
        end
    end

    best_snr_db = best_global_snr;
    if isempty(best_snr_db) || ~isfinite(best_snr_db)
        best_tx_id = NaN;
        best_snr_db = NaN;
        best_snr_sc_db = NaN(size(rx_sc,1),1);
        best_txafdem = [];
        best_ber_sc = NaN(size(rx_sc,1),1);
        align_info = struct('shift', NaN, 'rx_variant', 'none');
    end
end

function [snr_frame_db, snr_sc_db, txafdem_matrix, ber_sc] = compute_rx1_style_snr_local(rx_sc, tx_ref, C, DSP_template)
    rx_sc = double(rx_sc);
    tx_ref = double(tx_ref);
    n_sc = min(size(rx_sc,1), size(tx_ref,1));
    n_sym = min(size(rx_sc,2), size(tx_ref,2));
    rx_use = rx_sc(1:n_sc, 1:n_sym);
    tx_use = tx_ref(1:n_sc, 1:n_sym);

    txafdem_matrix = NaN(n_sc, n_sym);
    ber_sc = NaN(n_sc, 1);
    ok_row = false(n_sc, 1);

    for sc = 1:n_sc
        DSP = DSP_template;
        DSP.DEMAPPER.N0 = 0;
        try
            [DSP.DEMAPPER, txafdem] = symDemapper(rx_use(sc,:), tx_use(sc,:), C, DSP.DEMAPPER);
            txafdem_matrix(sc,:) = txafdem;
            ok_row(sc) = all(isfinite(real(txafdem))) && all(isfinite(imag(txafdem)));
            try
                [BER, ~] = BER_eval(DSP.DEMAPPER.txBits, DSP.DEMAPPER.rxBits);
                ber_sc(sc) = BER;
            catch
                ber_sc(sc) = NaN;
            end
        catch
            ok_row(sc) = false;
        end
    end

    snr_sc_db = NaN(n_sc, 1);
    valid_rows = find(ok_row);
    if isempty(valid_rows)
        snr_frame_db = NaN;
        return;
    end

    try
        [~, snr_tmp] = EVM_eval(rx_use(valid_rows,:), txafdem_matrix(valid_rows,:));
        snr_tmp = snr_tmp(:);
        n_fill = min(length(valid_rows), length(snr_tmp));
        snr_sc_db(valid_rows(1:n_fill)) = snr_tmp(1:n_fill);
    catch
        for ii = 1:length(valid_rows)
            sc = valid_rows(ii);
            try
                [~, s] = EVM_eval(rx_use(sc,:), txafdem_matrix(sc,:));
                snr_sc_db(sc) = s;
            catch
                snr_sc_db(sc) = NaN;
            end
        end
    end

    valid_snr = snr_sc_db(isfinite(snr_sc_db));
    if isempty(valid_snr)
        snr_frame_db = NaN;
    else
        snr_frame_db = 10 * log10(mean(10.^(valid_snr/10)));
    end
end

function tx_ref_out = normalize_tx_ref_shape_for_rx1_local(tx_ref, n_sc)
    x = tx_ref;
    if size(x,1) == n_sc
        tx_ref_out = x;
    elseif size(x,2) == n_sc
        tx_ref_out = x.';
    else
        x = x(:);
        n_sym = floor(length(x) / n_sc);
        if n_sym < 1, error('Invalid tx_ref shape'); end
        x = x(1:n_sc*n_sym);
        tx_ref_out = reshape(x, n_sc, n_sym);
    end
end

function [rx_use, tx_use] = align_rx_tx_by_symbol_shift_local(rx_sc, tx_ref, shift)
    rx = rx_sc;
    tx = tx_ref;
    n_sc = min(size(rx,1), size(tx,1));
    rx = rx(1:n_sc, :);
    tx = tx(1:n_sc, :);
    nr = size(rx,2);
    nt = size(tx,2);
    if shift >= 0
        r_start = 1 + shift;
        t_start = 1;
    else
        r_start = 1;
        t_start = 1 - shift;
    end
    n_sym = min(nr - r_start + 1, nt - t_start + 1);
    if n_sym < 20
        rx_use = [];
        tx_use = [];
        return;
    end
    rx_use = rx(:, r_start:r_start+n_sym-1);
    tx_use = tx(:, t_start:t_start+n_sym-1);
end

%% =====================================================================
%% CDM
%% =====================================================================
function cdm = make_cdm_from_rxsc_local(rx_sc, nbin, clip_val)
    z = rx_sc(:);
    z = z(isfinite(real(z)) & isfinite(imag(z)));
    if isempty(z)
        cdm = zeros(nbin, nbin);
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
end
