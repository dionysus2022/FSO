% =========================================================================
% rx_exp_cdm_dataset_generator.m - 实验信号帧级别 CDM 数据集生成器（仅处理 sub01）
% =========================================================================
% 遍历 5 种调制格式 × sub01 (25帧) = 125 个原始实验样本
% 清洗机制：自动拦截并剔除平均 SNR < 3 dB 的深衰落死帧，保证数据集纯净度
% 输出: dataset_cdm_exp/ 下 exp_frame_{t}_mod_{Label_Bits}.mat
% 每个文件包含: Distorted_CDM [64×64], Ideal_CDM [64×64], Label_Bits, 原始I/Q序列
% =========================================================================
clear; clear global; close all; clc;

%% 1. 路径与库配置
addpath(genpath('D:\matlab\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));
global PROG; PROG.showMessagesLevel = 2; initProg();

% 默认关闭循环内部的所有绘图弹窗，执行纯矩阵运算提速
set(0, 'DefaultFigureVisible', 'off');

data_root   = 'D:\matlab\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
rx_dir      = fullfile(data_root, 'rx_data');           
ref_dir     = fullfile(data_root, 'tx_1frame_5mod'); 
dataset_dir = fullfile(data_root, 'dataset_cdm_exp'); % 最终生成的实验训练集根目录
if ~exist(dataset_dir, 'dir'), mkdir(dataset_dir); end

%% 2. 循环边界与参数配置
mod_list = {'2QAM', '4QAM', '16QAM', '64QAM', '256QAM'}; 
% 对应仿真参考代码中的比特标签：2QAM->1, 4QAM->2, 16QAM->4, 64QAM->6, 256QAM->8
mod_bits = [1, 2, 4, 6, 8]; 

t_start  = 1; 
t_end    = 100;       % 严格限定处理第一个子文件夹 sub01 (1~25帧)
scope_Fs = 80e9;     % 示波器原生硬件采样率

%% 3. 基础通信与 OFDM 参数设置（固定不变）
SIG.symRate = 8e9; SIG.bitRate_net = 8e9;
SIG.modulation = 'QAM'; SIG.rollOff = 0.25; SIG.nPol = 1;
SIG.nSyms = 2^7; nSpS = 5; laserLW = 0e6;
FEC_rate = 1; pilotRate = 1; useCPE2 = false;
ofdm.NumberOfIFFTSamples = 256; 
ofdm.Carrier_location = [4:126];
ofdm.Carrier_location_demo = [4:126, 132:254];
ofdm.NumberOfCarriers = length(ofdm.Carrier_location);
ofdm.NumberOfCarriers_demo = length(ofdm.Carrier_location_demo);
ofdm.NumberOfGuardTime = 16; Fs = 10e9; Fg = 10e9;
nBpS_net = SIG.bitRate_net / (SIG.nPol * SIG.symRate * FEC_rate * pilotRate);

% 基础 DSP 配置（固定不变）
DSP.MF.type = 'RRC'; DSP.MF.rollOff = SIG.rollOff;
DSP.CPE1.method = 'pilot-based:optimized'; DSP.CPE1.decision = 'data-aided';
DSP.CPE1.nTaps_min = 1; DSP.CPE1.nTaps_max = 201; 
DSP.CPE2.method = 'BPS'; DSP.CPE2.nTaps = 22;
DSP.CPE2.nTaps_min = 1; DSP.CPE2.nTaps_max = 501;
DSP.CPE2.nTestPhases = 10; DSP.CPE2.angleInterval = pi/8;
DSP.DEMAPPER.normMethod = 'MMSE';

N_sc = ofdm.NumberOfCarriers;
total_saved_count = 0;

fprintf('\n🚀 开始提取实验信号帧级别 CDM 数据集（含深衰落清洗 + 独立 AGC）...\n');

%% ======================= 【外层大循环：遍历 5 种调制格式】 =======================
for m_idx = 1:length(mod_list)
    current_mod = mod_list{m_idx};
    Label_Bits  = mod_bits(m_idx); % 获取当前格式对齐仿真的标准比特标签
    
    fprintf('\n============== 正在处理调制格式: [%s] (Label_Bits = %d) ==============\n', current_mod, Label_Bits);
    
    % 动态更新判决边界与解调器阶数，防止高阶 QAM 假死
    current_M = str2double(regexp(current_mod, '\d+', 'match', 'once'));
    SIG.M = current_M;
    TX.SIG = setSignalParams('symRate', SIG.symRate, 'M', SIG.M, ...
        'nPol', SIG.nPol, 'nBpS', nBpS_net, 'nSyms', SIG.nSyms, ...
        'roll-off', SIG.rollOff, 'modulation', SIG.modulation);
    TX.QAM = QAM_config(TX.SIG);
    TX.BIT.source = 'randi'; TX.BIT.seed = 100;
    TX.PS.type = 'RRC'; TX.PS.rollOff = TX.SIG.rollOff; TX.PS.nTaps = 4096;
    TX.DAC.RESAMP.sampRate = nSpS * TX.SIG.symRate; TX.LASER.linewidth = laserLW;
    TX.PILOTS.active = true; TX.PILOTS.rate = pilotRate; TX.PILOTS.option = 'outerQPSK';
    TX.FEC.active = false; TX.FEC.rate = FEC_rate; TX.FEC.nIter = 50; TX.PCS.method = 'CCDM';
    ofdm.size = SIG.nSyms;
    C = TX.QAM.IQmap;
    DSP.CPE1.PILOTS = TX.PILOTS;
    
    % 锁定当前格式对应的 sub01 物理路径
    current_rx_dir  = fullfile(rx_dir, current_mod, '1');            % ...\rx_data\{mod}\1\
    current_ref_dir = fullfile(ref_dir, current_mod, 'sub01');       % ...\tx_1frame_5mod\{mod}\sub01\
    
    %% ======================= 【内层循环：处理 sub01 内的 1~25 帧】 =======================
    for t = t_start:t_end
        bin_file = fullfile(current_rx_dir, sprintf('%d.bin', t));
        mat_file = fullfile(current_ref_dir, sprintf('sig_%04d.mat', t));
        
        % 鲁棒性检查：跳过缺失文件
        if ~exist(bin_file, 'file') || ~exist(mat_file, 'file'), continue; end
        d = dir(bin_file); if d.bytes == 0, continue; end
        
        %% --- Step 1: 精准读取 Keysight 二进制流（完美原版原封不动） ---
        fid = fopen(bin_file, 'rb');
        if fid == -1, continue; end
        cookie=fread(fid,2,'*char')'; version=fread(fid,2,'*char')';
        file_size=fread(fid,1,'int32'); num_waveforms=fread(fid,1,'int32');
        header_size=fread(fid,1,'int32'); wave_type=fread(fid,1,'int32');
        num_buffers=fread(fid,1,'int32'); num_points=fread(fid,1,'int32');
        count=fread(fid,1,'int32'); x_disp_range=fread(fid,1,'float32');
        x_disp_orig=fread(fid,1,'float64'); x_inc=fread(fid,1,'float64');
        x_orig=fread(fid,1,'float64'); x_units=fread(fid,1,'int32');
        y_units=fread(fid,1,'int32'); date_str=fread(fid,16,'*char')';
        time_str=fread(fid,16,'*char')'; frame_str=fread(fid,24,'*char')';
        wave_str=fread(fid,16,'*char')'; time_tag=fread(fid,1,'float64');
        segment_index=fread(fid,1,'uint32'); data_header_size=fread(fid,1,'int32');
        buffer_type=fread(fid,1,'int16'); bytes_per_point=fread(fid,1,'int16');
        buffer_size=fread(fid,1,'int32');
        
        switch bytes_per_point
            case 4, OutputFSO = fread(fid, num_points, 'float32').';
            case 2, OutputFSO = fread(fid, num_points, 'int16').';
            case 1, OutputFSO = fread(fid, num_points, 'int8').';
            otherwise, OutputFSO = fread(fid, num_points, 'double').';
        end
        fclose(fid);
        
        %% --- Step 2: 前端 DSP 预处理、重采样与时域归一化 ---
        load(mat_file, 'data_tx');
        OutputFSO = resample(OutputFSO, 16e9, scope_Fs);
        OutputFSO = OutputFSO - mean(OutputFSO);
        AMP_rate2 = 1 / (sum(abs(OutputFSO)) / length(OutputFSO));
        data_in   = OutputFSO * AMP_rate2;
        
        %% --- Step 3: 原生 deOFDM 解调与基带矩阵变换 ---
        S.rx_1sps = data_in;
        S.rx_1sps = deOFDM(S.rx_1sps, ofdm, SIG.nSyms);
        if pilotRate < 1, [S.rx_1sps, DSP.CPE1] = carrierPhaseEstimation(S.rx_1sps, S.tx, DSP.CPE1); end
        if useCPE2, [S.rx_1sps, DSP.CPE2] = carrierPhaseEstimation(S.rx_1sps, S.tx, DSP.CPE2, C); end
        if pilotRate < 1, [S.rx_1sps, S.tx] = pilotSymbols_rmv(S.rx_1sps, S.tx, DSP.CPE1.PILOTS); end
        
        S.tx = data_tx.';
        S.rx_1sps = reshape(S.rx_1sps, SIG.nSyms, ofdm.NumberOfCarriers_demo).';
        
        %% --- Step 4: 判决与等效平均 SNR 测算（拦截清洗核心） ---
        S.txafdem_matrix = zeros(123, SIG.nSyms);
        for i = 1:123
            [DSP.DEMAPPER, S.txafdem] = symDemapper(S.rx_1sps(i,:), S.tx(i,:), C, DSP.DEMAPPER);
            DSP.DEMAPPER.N0 = 0;
            S.txafdem_matrix(i,:) = S.txafdem;
        end
        
        S.rx_1sps_eval = S.rx_1sps(1:123, :);
        [~, SNR_CAL] = EVM_eval(S.rx_1sps_eval, S.txafdem_matrix);
        
        v = SNR_CAL(SNR_CAL > 0 & isfinite(SNR_CAL));
        avg_SNR = 10 * log10(mean(10.^(v / 10)));
        
        % 数据清洗拦截机制：低于解调判定门限的死帧绝不计入特征训练集
        if avg_SNR < 3 || isnan(avg_SNR)
            fprintf('  [Cleaned] File %02d.bin 遭遇深衰落 (%.2f dB)，已被拦截剔除。\n', t, avg_SNR);
            continue;
        end
        
        %% --- Step 5: 🌟帧级别特征降维与融合（无缝接入 CDM 制造流） ---
        % 提取通过解调补偿后的真实接收星座复数信号与理想发射信号
        % 提取前 123 个有效子载波数据
        rx_block = S.rx_1sps(1:123, :);
        tx_block = S.tx(1:123, :);
        
        % 核心动作：将 123 个子载波的所有符号全部展平为帧级别全局长向量
        rx_symbols_flat = rx_block(:);
        tx_symbols_flat = tx_block(:);
        
        %% --- Step 6: 独立 AGC 归一化（接收与发射各自 RMS 归一化，完全同步仿真逻辑） ---
        rx_norm_factor = sqrt(mean(abs(rx_symbols_flat).^2));
        rx_symbols_flat = rx_symbols_flat / rx_norm_factor;
        
        tx_norm_factor = sqrt(mean(abs(tx_symbols_flat).^2));
        tx_symbols_flat = tx_symbols_flat / tx_norm_factor;
        
        %% --- Step 7: 生成高斯平滑 CDM 矩阵 ---
        Distorted_CDM = generate_CDM_Smooth(rx_symbols_flat, 64);
        Ideal_CDM     = generate_CDM_Smooth(tx_symbols_flat, 64);
        
        %% --- Step 8: 离线特征序列打包固化持久化保存 ---
        % 文件名格式对齐：exp_frame_{帧号}_mod_{比特标签}.mat
        save_name = fullfile(dataset_dir, sprintf('exp_frame_%02d_mod_%d.mat', t, Label_Bits));
        
        % 拆分实部与虚部，生成标准的二维 I/Q 物理流，供后续时域一维网络备用
        rx_IQ = [real(rx_symbols_flat), imag(rx_symbols_flat)];
        tx_IQ = [real(tx_symbols_flat), imag(tx_symbols_flat)];
        
        save(save_name, 'Distorted_CDM', 'Ideal_CDM', 'Label_Bits', ...
            'rx_symbols_flat', 'tx_symbols_flat', 'rx_IQ', 'tx_IQ');
        
        total_saved_count = total_saved_count + 1;
        fprintf('  [Exported] File %02d.bin -> CDM 特征图创建完毕 (SNR=%.2f dB)\n', t, avg_SNR);
    end
end

%% 4. 全局报告输出
set(0, 'DefaultFigureVisible', 'on'); % 还原绘图弹窗设置
fprintf('\n==================================================================\n');
fprintf('  🎉 实验信号帧级别特征集构建圆满成功！\n');
fprintf('  累计捕获有效幸存样本：%d 个 (剔除了被强湍流吞噬的死帧)\n', total_saved_count);
fprintf('  实验 CDM 数据集存储盘符路径: %s\n', dataset_dir);
fprintf('==================================================================\n');

%% ======================= 【附带辅助函数：二维直方图+高斯平滑】 =======================
function cdm = generate_CDM_Smooth(complex_symbols, grid_size)
    % 设立复平面判定边界 [-2.0, 2.0]
    edges = linspace(-2.0, 2.0, grid_size + 1);
    % 二维网格统计密度计数
    [N, ~, ~] = histcounts2(real(complex_symbols), imag(complex_symbols), edges, edges);
    % 旋转对齐坐标轴
    cdm = rot90(N);
    % 施加高斯模糊滤波器平滑边缘，消除空洞，提高卷积神经网络特征识别率
    cdm = imgaussfilt(cdm, 1.0);
    % 最大值峰值归一化
    if max(cdm(:)) > 0, cdm = cdm / max(cdm(:)); end
end