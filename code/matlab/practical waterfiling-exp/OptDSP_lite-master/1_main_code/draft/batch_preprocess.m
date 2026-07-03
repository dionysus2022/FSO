% batch_preprocess.m
% 按照 预处理流程.txt 批量处理所有接收信号
% 输入: rx_data/2026.06.26/<mod>/<sub>/<idx>.bin
% 输出: dataset_lightprior/
%          full_frame_16G/<mod>/<sub>/sig_<XXXX>_frame<k>.mat
%          time_32768/<mod>/<sub>/sig_<XXXX>_frame<k>.mat
%        index.csv

clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));
global PROG; PROG.showMessagesLevel = 1; initProg();

%% ===================== 配置 =====================
data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
rx_date   = '2026.06.26';
tx_root   = fullfile(data_root, 'tx_3frame_6mod');
out_root  = fullfile(data_root, 'dataset_lightprior');

Fs_rx   = 80e9;   % 示波器采样率
Fs_base = 16e9;   % 基带采样率

mod_names = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};
n_frames = 3;
M_time = 32768;    % 时域识别固定长度

% 轻先验同步参数
sync_decim = 20;   % 包络相关降采样倍数

%% ===================== 输出目录 =====================
out_full = fullfile(out_root, 'full_frame_16G');
out_time = fullfile(out_root, sprintf('time_%d', M_time));
mkdir(out_full); mkdir(out_time);

index_file = fullfile(out_root, 'index.csv');
fid_csv = fopen(index_file, 'w');
% CSV header
    fprintf(fid_csv, 'out_file,label_id,label_name,mod_order,file_id,sig_idx,frame_idx,tx_frame_id,sub_name,turbulence,sync_metric,valid_flag\n');

total_files = 0;
total_frames = 0;

%% ===================== 遍历所有调制格式 =====================
for m = 1:length(mod_names)
    mod_name = mod_names{m};
    label_id = m - 1;
    
    % 调制阶数
    switch mod_name
        case 'QPSK',   M = 4;
        case '16QAM',  M = 16;
        case '32QAM',  M = 32;
        case '64QAM',  M = 64;
        case '128QAM', M = 128;
        case '256QAM', M = 256;
    end
    
    rx_dir = fullfile(data_root, 'rx_data', rx_date, mod_name);
    tx_dir = fullfile(tx_root, mod_name);
    
    if ~exist(rx_dir, 'dir')
        warning('RX目录不存在: %s，跳过', rx_dir);
        continue;
    end
    
    % 遍历 sub1~sub4
    sub_dirs = dir(fullfile(rx_dir, 'sub*'));
    % 湍流映射: sub1=weak, sub2=moderate, sub3=strong, sub4=strong
    sub_turb_map = containers.Map();
    sub_turb_map('sub1') = 'weak';
    sub_turb_map('sub2') = 'moderate';
    sub_turb_map('sub3') = 'strong';
    sub_turb_map('sub4') = 'strong';
    
    for s = 1:length(sub_dirs)
        sub_name = sub_dirs(s).name;
        bin_files = dir(fullfile(rx_dir, sub_name, '*.bin'));
        
        if isempty(bin_files)
            continue;
        end
        
        % 创建输出子目录
            turb_name = sub_turb_map(sub_name);
            % 创建输出子目录（用湍流名）
            mkdir(fullfile(out_full, mod_name, turb_name));
            mkdir(fullfile(out_time, mod_name, turb_name));
        
        for b = 1:length(bin_files)
            % 提取文件编号
            [~, fname] = fileparts(bin_files(b).name);
            sig_idx = str2double(fname);
            if isnan(sig_idx) || sig_idx < 1
                continue;
            end
            
            rx_bin = fullfile(rx_dir, sub_name, bin_files(b).name);
            tx_txt = fullfile(tx_dir, sub_name, sprintf('sig_%04d.txt', sig_idx));
            
            total_files = total_files + 1;
            
            %% Step 1: 读取80G长序列
            try
                rx80 = read_keysight_bin_local(rx_bin);
            catch ME
                warning('读取失败: %s — %s', rx_bin, ME.message);
                continue;
            end
            rx80 = rx80(:);
            
            %% Step 2: 轻先验预处理
            rx80 = rx80 - mean(rx80);
            rx80 = rx80 ./ (rms(rx80) + eps);
            
            %% Step 3: 确定帧长
            if exist(tx_txt, 'file')
                tx_ref = load_ascii_complex_local(tx_txt);
                frame_len_80 = floor(length(tx_ref(:)) / n_frames);
            else
                % 理论估计
                payload_16 = (256 + 16) * 128;
                frame_16 = 80 + 16 + 256*2 + payload_16;
                frame_len_80 = ceil(frame_16 * Fs_rx / Fs_base / 256) * 256;
            end
            
            if frame_len_80 <= 0
                warning('帧长无效: %s', rx_bin);
                continue;
            end
            
            %% Step 4: 轻先验同步（包络相关，不用TX波形模板）
            % 用帧长度级别的包络相关做粗定位
            decim = sync_decim;
            rx_env = abs(rx80(1:decim:end));
            rx_env = rx_env - mean(rx_env);
            rx_env = rx_env ./ (std(rx_env) + eps);
            
            % 构造帧周期模板（方波：帧数据区高、帧头间隙低）
            templ_len = round(frame_len_80 / decim);
            templ = ones(templ_len, 1);
            gap_len = round(80 / decim);  % 80个零的帧头
            templ(1:gap_len) = 0;
            
            % 滑动相关
            if length(rx_env) < length(templ)
                warning('信号太短: %s', rx_bin);
                continue;
            end
            c = conv(rx_env, flipud(templ), 'valid');
            [~, idx_max] = max(abs(c));
            rx_start_80 = (idx_max - 1) * decim + 1;
            sync_metric = max(abs(c)) / length(templ);
            
            % 循环修正：如果放不下3帧则前移1帧
            while rx_start_80 + n_frames * frame_len_80 - 1 > length(rx80)
                rx_start_80 = rx_start_80 - frame_len_80;
            end
            while rx_start_80 < 1
                rx_start_80 = rx_start_80 + frame_len_80;
            end
            
            %% Step 5: 切3帧 + 逐帧处理
            for k = 1:n_frames
                seg_start = rx_start_80 + (k-1) * frame_len_80;
                seg_end   = seg_start + frame_len_80 - 1;
                
                % 跨边界拼接（AWG循环）
                if seg_end <= length(rx80)
                    rx_frame_80 = rx80(seg_start:seg_end);
                else
                    part1 = rx80(seg_start:end);
                    part2 = rx80(1:(seg_end - length(rx80)));
                    rx_frame_80 = [part1; part2];
                end
                
                % 单帧resample到16G
                rx_frame_16 = resample(rx_frame_80.', Fs_base, Fs_rx).';
                rx_frame_16 = rx_frame_16 - mean(rx_frame_16);
                rx_frame_16 = rx_frame_16 ./ (rms(rx_frame_16) + eps);
                
                % 提取时域净载荷（跳过帧头+训练段，仅OFDM数据部分）
                header_len = 80 + 16 + 256*2;  % zeros + CP + 2×LTS
                header_len_16 = round(header_len * Fs_base / Fs_rx);
                payload_start = header_len_16 + 1;
                
                if payload_start + M_time - 1 <= length(rx_frame_16)
                    rx_time = rx_frame_16(payload_start : payload_start + M_time - 1);
                else
                    rx_time = rx_frame_16(payload_start : end);
                    if length(rx_time) < M_time
                        rx_time = [rx_time; zeros(M_time - length(rx_time), 1)];
                    end
                end
                
                %% 构建样本结构
                file_id = sprintf('%s_%s_sig%04d', mod_name, sub_name, sig_idx);
                
                sample = struct();
                sample.rx_frame_16_full = single(rx_frame_16);
                sample.rx_time = single(rx_time);
                sample.label_id = label_id;
                sample.label_name = mod_name;
                sample.mod_order = M;
                sample.file_id = file_id;
                sample.sig_idx = sig_idx;
                sample.sub_name = sub_name;
                sample.turbulence = turb_name;
                sample.frame_idx = k;
                sample.tx_frame_id = k;
                sample.rx_bin_file = rx_bin;
                sample.rx_start_80 = rx_start_80;
                sample.seg_start_80 = seg_start;
                sample.seg_end_80 = seg_end;
                sample.frame_len_80 = frame_len_80;
                sample.Fs_rx = Fs_rx;
                sample.Fs_base = Fs_base;
                sample.M = M_time;
                sample.sync_method = 'light_prior_frame_sync';
                sample.sync_metric = sync_metric;
                sample.valid_flag = true;
                
                % 保存完整帧版本
                % 保存完整帧版本（用湍流名）
                out_file_full = fullfile(out_full, mod_name, turb_name, ...
                    sprintf('sig_%04d_frame%d.mat', sig_idx, k));
                save(out_file_full, 'sample', '-v7.3');
                
                % 保存时域版本（只保留rx_time减小体积）
                sample_time = rmfield(sample, 'rx_frame_16_full');
                % 保存时域版本（用湍流名）
                out_file_time = fullfile(out_time, mod_name, turb_name, ...
                    sprintf('sig_%04d_frame%d.mat', sig_idx, k));
                save(out_file_time, 'sample_time', '-v7.3');
                
                % CSV索引
                fprintf(fid_csv, '%s,%d,%s,%d,%s,%d,%d,%d,%s,%s,%.4f,%d\n', ...
                    out_file_full, label_id, mod_name, M, file_id, ...
                    sig_idx, k, k, sub_name, turb_name, sync_metric, 1);
                
                total_frames = total_frames + 1;
            end
            
            if mod(b, 5) == 0
            fprintf('[%s/%s->%s] %d/%d\n', mod_name, sub_name, turb_name, b, length(bin_files));
            end
        end
    end
    fprintf('=== %s 完成 ===\n', mod_name);
end

fclose(fid_csv);

fprintf('\n========================================\n');
fprintf('预处理完成!\n');
fprintf('处理文件: %d\n', total_files);
fprintf('生成样本: %d (3帧/文件)\n', total_frames);
fprintf('完整帧:   %s\n', out_full);
fprintf('时域数据: %s\n', out_time);
fprintf('索引文件: %s\n', index_file);
fprintf('========================================\n');

%% ===================== 局部函数 =====================
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
        case 4, y = fread(fid, num_points, 'float32').';
        case 2, y = fread(fid, num_points, 'int16').';
        case 1, y = fread(fid, num_points, 'int8').';
        otherwise, y = fread(fid, num_points, 'double').';
    end
    fclose(fid);
end

function x = load_ascii_complex_local(filename)
    tmp = load(filename);
    if size(tmp, 2) >= 2
        x = complex(tmp(:,1), tmp(:,2));
    else
        x = tmp(:);
    end
end
