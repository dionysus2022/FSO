% batch_preprocess_v2.m
% 改进版预处理：先用 rx_exp_weak.m 的同步算法找帧头
% → 验证帧结构 → 切前3帧 → 保存 + 解调 + SNR
%
% 同步: packet_edge_power_dect + rx_fine_time_sync_cross_corr
% 解调: 手动解调（LTS+CFO+FFT+均衡）+ symDemapper + EVM_eval
% 参考: tx_3frame_6mod/<mod>/<sub>/sig_XXXX_frameY.mat (data_tx)

clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));
global PROG; PROG.showMessagesLevel = 1; initProg(); co = 1;

%% ===================== 配置 =====================
data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
rx_date   = '2026.06.26';
tx_root   = fullfile(data_root, 'tx_3frame_6mod');
out_root  = fullfile(data_root, 'dataset_v2');

Fs_rx   = 80e9;
Fs_base = 16e9;

mod_names = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};
n_frames = 3;
M_time = 32768;

% OFDM 参数
ofdm.NumberOfIFFTSamples = 256;
ofdm.NumberOfGuardTime = 16;
ofdm.Carrier_location = 4:126;
ofdm.Carrier_location_demo = [4:126, 132:254];
ofdm.NumberOfCarriers = length(ofdm.Carrier_location);
ofdm.NumberOfCarriers_demo = length(ofdm.Carrier_location_demo);
ofdm.size = 128;

% TX 参数模板
SIG.nPol = 1; SIG.nSyms = 128; SIG.rollOff = 0.25; SIG.modulation = 'QAM';

% 湍流映射
turb_map = containers.Map('KeyType','char','ValueType','char');
turb_map('sub1') = 'weak';    turb_map('sub2') = 'moderate';
turb_map('sub3') = 'strong';  turb_map('sub4') = 'strong';

%% ===================== 输出目录 =====================
out_full = fullfile(out_root, 'full_frame_16G');
out_time = fullfile(out_root, sprintf('time_%d', M_time));
out_snr  = fullfile(out_root, 'snr_results');
mkdir(out_full); mkdir(out_time); mkdir(out_snr);

% CSV索引
fid_csv = fopen(fullfile(out_root, 'index.csv'), 'w');
fprintf(fid_csv, 'out_file,label_id,label_name,mod_order,file_id,sig_idx,frame_idx,turbulence,sync_metric,snr_mean_db,valid_flag\n');

total_files = 0; total_frames = 0;
total_demod_ok = 0; total_demod_fail = 0;
skip_no_ref = 0; skip_short = 0; skip_sync = 0;

%% ===================== 主循环 =====================
for m = 1:length(mod_names)
    mod_name = mod_names{m};
    label_id = m - 1;
    
    switch mod_name
        case 'QPSK',   Mq = 4;  bits = 2;
        case '16QAM',  Mq = 16; bits = 4;
        case '32QAM',  Mq = 32; bits = 5;
        case '64QAM',  Mq = 64; bits = 6;
        case '128QAM', Mq = 128;bits = 7;
        case '256QAM', Mq = 256;bits = 8;
    end
    nBpS_net = bits - 0.2*(bits > 2);
    TX.SIG = setSignalParams('symRate',8e9,'M',Mq,'nPol',1,...
        'nBpS',nBpS_net,'nSyms',128,'roll-off',0.25,'modulation','QAM');
    TX.QAM = QAM_config(TX.SIG);
    C = TX.QAM.IQmap;
    DSP.DEMAPPER.normMethod = 'MMSE';
    
    rx_dir = fullfile(data_root, 'rx_data', rx_date, mod_name);
    if ~exist(rx_dir,'dir'), continue; end
    
    % 只处理 sub1~sub3
    sub_list = {'sub1','sub2','sub3'};
    for s = 1:length(sub_list)
        sub_name = sub_list{s};
        turb_name = turb_map(sub_name);
        bin_list = dir(fullfile(rx_dir, sub_name, '*.bin'));
        if isempty(bin_list), continue; end
        
        % 预处理：先读全部.bin? 太多，逐文件处理
        % 每文件: 读80G → 重采样16G → 同步 → 切3帧 → 保存+解调
        
        mkdir(fullfile(out_full, mod_name, turb_name));
        mkdir(fullfile(out_time, mod_name, turb_name));
        
        for b = 1:length(bin_list)
            [~, fname] = fileparts(bin_list(b).name);
            sig_idx = str2double(fname);
            if isnan(sig_idx) || sig_idx < 1, continue; end
            
            rx_bin = fullfile(rx_dir, sub_name, bin_list(b).name);
            total_files = total_files + 1;
            
            %% === A. 读取80G + 预处理 ===
            try
                rx80 = read_keysight_bin_local(rx_bin);
            catch
                warning('读文件失败: %s', rx_bin); continue;
            end
            rx80 = rx80(:).';
            rx80 = rx80 - mean(rx80);
            rx80 = rx80 ./ (rms(rx80) + eps);
            
            %% === B. 重采样到16G ===
            rx16 = resample(rx80, Fs_base, Fs_rx);
            rx16 = rx16 - mean(rx16);
            rx16 = rx16 ./ (rms(rx16) + eps);
            
            %% === C. 同步：用 rx_exp_weak.m 一样的算法 ===
            % packet_edge_power_dect: 找80零帧头
            zeros_head = 80;
            n_guard = ofdm.NumberOfGuardTime;  % 16
            n_fft = ofdm.NumberOfIFFTSamples;  % 256
            sym_len = n_fft + n_guard;          % 272
            symbol_bits = zeros_head + n_guard + n_fft*2 + sym_len * 128;  % 35424
            
            % 确保有足够信号做同步
            if length(rx16) < 2*symbol_bits
                % 信号太短，跳过
                continue;
            end
            
            % 粗同步：找80零（最小功率窗）
            [detected_packet, edge_index] = packet_edge_power_dect(...
                rx16(1:2*symbol_bits), zeros_head);
            
            % 细同步：LTS互相关
            load('LongTrainSym_ini.mat', 'LongTrainSym_ini');
            LTS_f = LongTrainSym_ini(1:n_fft);
            LTS_f([1 n_fft/2+1]) = 0;
            ltrs_in = LTS_f;
            ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));
            
            [fine_time_est, ~, ~] = rx_fine_time_sync_cross_corr(...
                detected_packet, n_guard, ltrs_in, zeros_head, 0);
            
            % 帧起始精确位置（16G采样率）
            frame_start_16 = edge_index + fine_time_est - 1;
            sync_metric = 1;  % 简化
            
            %% === D. 验证帧结构：检查是否有3个完整帧 ===
            % 从 frame_start_16 开始，检查能否放下至少 n_frames 帧
            available_len = length(rx16) - frame_start_16 + 1;
            max_frames = floor(available_len / symbol_bits);
            if max_frames < n_frames
                % 信号太短，只处理能放下的帧
                n_actual = max_frames;
            else
                n_actual = n_frames;
            end
            
            %% === E. 切前3帧 + 保存 + 解调 ===
            for k = 1:n_actual
                seg_start = frame_start_16 + (k-1) * symbol_bits;
                seg_end   = seg_start + symbol_bits - 1;
                
                if seg_end > length(rx16)
                    break;  % 超出信号范围
                end
                
                rx_frame = rx16(seg_start:seg_end);  % 1帧16G信号
                
                %% E1. 保存完整帧
                file_id = sprintf('%s_%s_sig%04d', mod_name, sub_name, sig_idx);
                sample = struct();
                sample.rx_frame_16_full = single(rx_frame);
                sample.label_id = label_id;
                sample.label_name = mod_name;
                sample.mod_order = Mq;
                sample.turbulence = turb_name;
                sample.file_id = file_id;
                sample.sig_idx = sig_idx;
                sample.sub_name = sub_name;
                sample.frame_idx = k;
                sample.rx_bin_file = rx_bin;
                sample.Fs_rx = Fs_rx;
                sample.Fs_base = Fs_base;
                sample.sync_method = 'deofdm_sync';
                sample.sync_metric = sync_metric;
                sample.valid_flag = true;
                
                out_full_file = fullfile(out_full, mod_name, turb_name, ...
                    sprintf('sig_%04d_frame%d.mat', sig_idx, k));
                save(out_full_file, 'sample', '-v7.3');
                
                %% E2. 保存时域净载荷（32768点）
                header_len = zeros_head + n_guard + 2*n_fft;  % 80+16+512=608
                header_len_16 = round(header_len * Fs_base / Fs_rx);
                payload_start = header_len_16 + 1;
                
                if payload_start + M_time - 1 <= length(rx_frame)
                    rx_time = rx_frame(payload_start : payload_start + M_time - 1);
                else
                    rx_time = rx_frame(payload_start : end);
                    if length(rx_time) < M_time
                        rx_time = [rx_time, zeros(1, M_time - length(rx_time))];
                    end
                end
                
                sample_time = rmfield(sample, 'rx_frame_16_full');
                sample_time.rx_time = single(rx_time);
                out_time_file = fullfile(out_time, mod_name, turb_name, ...
                    sprintf('sig_%04d_frame%d.mat', sig_idx, k));
                save(out_time_file, 'sample_time', '-v7.3');
                
                %% E3. 解调 + SNR（用tx参考）
                % 加载参考
                tx_sub = sub_name;
                ref_file = fullfile(tx_root, mod_name, tx_sub, ...
                    sprintf('sig_%04d_frame%d.mat', sig_idx, k));
                if ~exist(ref_file, 'file')
                    % csv写默认值
                    skip_no_ref = skip_no_ref + 1;
                    fprintf(fid_csv, '%s,%d,%s,%d,%s,%d,%d,%s,%.4f,NaN,%d\n', ...
                        out_full_file, label_id, mod_name, Mq, file_id, ...
                        sig_idx, k, turb_name, sync_metric, 0);
                    continue;
                end
                ref = load(ref_file);
                if ~isfield(ref, 'data_tx'), continue; end
                tx_ref = ref.data_tx.';  % (123, 128)
                
                % 手动解调（避免deOFDM的2帧搜索要求）
                % 本地LTS
                load('LongTrainSym_ini.mat', 'LongTrainSym_ini');
                LTS_f2 = LongTrainSym_ini(1:n_fft);
                LTS_f2([1 n_fft/2+1]) = 0;
                ltrs_in2 = LTS_f2;
                ltrs_in2(1, n_fft/2+2:n_fft) = conj(ltrs_in2(1, n_fft/2:-1:2));
                LTS_t = ifft(ltrs_in2);
                
                % LTS相关（在帧内重新找精确位置）
                rx_r = rx_frame(:).';
                xc_len = length(rx_r) - n_fft + 1;
                xc = zeros(1, xc_len);
                for ni = 1:xc_len
                    xc(ni) = abs(sum(rx_r(ni:ni+n_fft-1) .* conj(LTS_t)));
                end
                [~, lts_pk] = max(xc);
                frm_st = max(1, lts_pk - zeros_head - n_guard - 1);
                
                % 取完整帧
                frm_en = min(length(rx_r), frm_st + symbol_bits - 1);
                if frm_en - frm_st + 1 < symbol_bits
                    skip_short = skip_short + 1;
                    continue;
                end
                rx_f = rx_r(frm_st:frm_en);
                
                % LTS + CFO
                lts1 = rx_f(zeros_head+n_guard+1 : zeros_head+n_guard+n_fft);
                lts2 = rx_f(zeros_head+n_guard+n_fft+1 : zeros_head+n_guard+2*n_fft);
                pd = angle(sum(lts1(:).*conj(lts2(:))));
                cfo = pd/(2*pi*n_fft);
                rx_f = rx_f .* exp(-1j*2*pi*cfo*(0:length(rx_f)-1)/n_fft);
                
                % 数据段
                ds = zeros_head + n_guard + 2*n_fft + 1;
                dp = rx_f(ds:end);
                nd = floor(length(dp)/sym_len);
                dp = dp(1:nd*sym_len);
                dm = reshape(dp, sym_len, nd);
                dn = dm(n_guard+1:end, :);
                fd = fft(dn, n_fft, 1)/sqrt(n_fft);
                
                % 信道估计
                lts_avg = (lts1(:)+lts2(:))/2;
                lts_fd = fft(lts_avg, n_fft)/sqrt(n_fft);
                Hch = lts_fd./(LTS_f2(:)+1e-12);
                Hch(abs(LTS_f2(:))<0.5) = 1;
                feq = fd./Hch;
                
                % 提取子载波
                carrier_loc = 4:126;
                n_sc = length(carrier_loc);
                rx_sc = feq(carrier_loc, :);  % (123, nd)
                
                % symDemapper + EVM_eval
                n_sym = min(size(rx_sc,2), size(tx_ref,2));
                rx_sc = rx_sc(:, 1:n_sym);
                tx_ref = tx_ref(:, 1:n_sym);
                
                txafdem = zeros(n_sc, n_sym);
                for sc = 1:n_sc
                    DSP.DEMAPPER.N0 = 0;
                    [DSP.DEMAPPER, td] = symDemapper(rx_sc(sc,:), tx_ref(sc,:), C, DSP.DEMAPPER);
                    txafdem(sc,:) = td;
                end
                [~, SNR_sc] = EVM_eval(rx_sc, txafdem);
                
                valid_snr = SNR_sc(isfinite(SNR_sc) & SNR_sc > 0);
                if ~isempty(valid_snr)
                    snr_mean = 10*log10(mean(10.^(valid_snr/10)));
                    total_demod_ok = total_demod_ok + 1;
                else
                    snr_mean = NaN;
                    total_demod_fail = total_demod_fail + 1;
                end
                
                % CSV
                fprintf(fid_csv, '%s,%d,%s,%d,%s,%d,%d,%s,%.4f,%.2f,%d\n', ...
                    out_full_file, label_id, mod_name, Mq, file_id, ...
                    sig_idx, k, turb_name, sync_metric, snr_mean, 1);
                
                total_frames = total_frames + 1;
            end
            
            if mod(b, 10) == 0
                fprintf('[%s/%s] %d/%d\n', mod_name, sub_name, b, length(bin_list));
            end
        end
    end
    fprintf('=== %s 完成 (%d帧) ===\n', mod_name, total_frames);
end

fclose(fid_csv);
fprintf('\n========================================\n');
fprintf('全部完成: %d文件, %d帧\n', total_files, total_frames);
fprintf('解调成功: %d帧\n', total_demod_ok);
fprintf('解调失败(SNR无效): %d帧\n', total_demod_fail);
fprintf('跳过(无参考): %d帧\n', skip_no_ref);
fprintf('跳过(帧不完整): %d帧\n', skip_short);
fprintf('成功率: %.1f%%\n', 100*total_demod_ok/max(total_demod_ok+total_demod_fail+skip_no_ref+skip_short,1));
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
