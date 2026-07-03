% =========================================================================
% rx_exp_cdm_dataset_generator_all.m - 全实验信号帧CDM数据集生成脚本
% =========================================================================
% 遍历5种调制格式 × 4个子文件夹，每组25帧 = 共500份原始实验样本
% 清洗机制：自动过滤并剔除平均SNR < 3 dB的深衰落失效帧，保证数据集纯净度
% 输出: dataset_cdm_exp_all/ 目录下 exp_frame_{1~100}_mod_{Label_Bits}.mat
% =========================================================================
clear; clear global; close all; clc;

%% 1. 路径与库配置
addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));
global PROG; PROG.showMessagesLevel = 2; initProg();

set(0, 'DefaultFigureVisible', 'off');

data_root   = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
rx_dir      = fullfile(data_root, 'rx_data');           
ref_dir     = fullfile(data_root, 'tx_1frame_5mod'); 
dataset_root = fullfile(data_root, 'dataset_cdm'); % 数据集根目录
weak_dir    = fullfile(dataset_root, 'weak');        % 弱湍流（文件夹1、4）
strong_dir  = fullfile(dataset_root, 'strong');      % 强湍流（文件夹2、3）
if ~exist(weak_dir, 'dir'),   mkdir(weak_dir);   end
if ~exist(strong_dir, 'dir'), mkdir(strong_dir); end

%% 2. 循环边界与参数配置
mod_list = {'2QAM', '4QAM', '16QAM', '64QAM', '256QAM'}; 
mod_bits = [1, 2, 4, 6, 8]; 

% 修改点1：新增子文件夹遍历列表
sub_list = {'1', '2', '3', '4'}; 

t_start  = 1; 
t_end    = 25;       % 单个子文件夹内包含25帧
scope_Fs = 80e9;     

%% 3. 基础通信与OFDM参数设置（固定不变）
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

DSP.MF.type = 'RRC'; DSP.MF.rollOff = SIG.rollOff;
DSP.CPE1.method = 'pilot-based:optimized'; DSP.CPE1.decision = 'data-aided';
DSP.CPE1.nTaps_min = 1; DSP.CPE1.nTaps_max = 201; 
DSP.CPE2.method = 'BPS'; DSP.CPE2.nTaps = 22;
DSP.CPE2.nTaps_min = 1; DSP.CPE2.nTaps_max = 501;
DSP.CPE2.nTestPhases = 10; DSP.CPE2.angleInterval = pi/8;
DSP.DEMAPPER.normMethod = 'MMSE';

N_sc = ofdm.NumberOfCarriers;
total_saved_count = 0;

fprintf('\n🚀 开始提取全部500帧实验信号CDM数据集（包含深衰落清洗 + 独立AGC归一化）...\n');

%% ======================= 【外层大循环：遍历5种调制格式】=======================
for m_idx = 1:length(mod_list)
    current_mod = mod_list{m_idx};
    Label_Bits  = mod_bits(m_idx); 
    
    fprintf('\n============== 正在处理调制格式: [%s] (Label_Bits = %d) ==============\n', current_mod, Label_Bits);
    
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
    
    %% ======================= 【修改点2：新增第二层循环，遍历4个子文件夹】=======================
    for s_idx = 1:length(sub_list)
        current_sub = sub_list{s_idx};
        sub_num = str2double(current_sub);
        
        % 动态拼接当前子文件夹路径
        current_rx_dir  = fullfile(rx_dir, current_mod, current_sub);            
        current_ref_dir = fullfile(ref_dir, current_mod, sprintf('sub%02d', sub_num));       
        
        %% ======================= 【内层循环：处理单个文件夹内1~25帧】=======================
        for t = t_start:t_end
            
            % 修改点3：计算全局帧编号 (1至100)，用于无重复文件命名
             global_t_pos = (sub_num - 1) * t_end + t;
            
            bin_file = fullfile(current_rx_dir, sprintf('%d.bin', global_t_pos));
            mat_file = fullfile(current_ref_dir, sprintf('sig_%04d.mat', global_t_pos));
            
            if ~exist(bin_file, 'file') || ~exist(mat_file, 'file'), continue; end
            d = dir(bin_file); if d.bytes == 0, continue; end
            
            %% --- Step 1: 精准读取Keysight二进制码流（数据处理代码完全不变） ---
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
            
            %% --- Step 2 ~ Step 7 完全不变的数据处理流程 ---
            load(mat_file, 'data_tx');
            OutputFSO = resample(OutputFSO, 16e9, scope_Fs);
            OutputFSO = OutputFSO - mean(OutputFSO);
            AMP_rate2 = 1 / (sum(abs(OutputFSO)) / length(OutputFSO));
            data_in   = OutputFSO * AMP_rate2;
            
            S.rx_1sps = data_in;
            S.rx_1sps = deOFDM(S.rx_1sps, ofdm, SIG.nSyms);
            if pilotRate < 1, [S.rx_1sps, DSP.CPE1] = carrierPhaseEstimation(S.rx_1sps, S.tx, DSP.CPE1); end
            if useCPE2, [S.rx_1sps, DSP.CPE2] = carrierPhaseEstimation(S.rx_1sps, S.tx, DSP.CPE2, C); end
            if pilotRate < 1, [S.rx_1sps, S.tx] = pilotSymbols_rmv(S.rx_1sps, S.tx, DSP.CPE1.PILOTS); end
            
            S.tx = data_tx.';
            S.rx_1sps = reshape(S.rx_1sps, SIG.nSyms, ofdm.NumberOfCarriers_demo).';
            
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
            
            if avg_SNR < 3 || isnan(avg_SNR)
                fprintf('  [Cleaned] %s-Sub%02d File %02d 遭遇深衰落(%.2f dB)，已过滤剔除\n', current_mod, sub_num, t, avg_SNR);
                continue;
            end
            
            rx_block = S.rx_1sps(1:123, :);
            tx_block = S.tx(1:123, :);
            rx_symbols_flat = rx_block(:);
            tx_symbols_flat = tx_block(:);
            
            rx_norm_factor = sqrt(mean(abs(rx_symbols_flat).^2));
            rx_symbols_flat = rx_symbols_flat / rx_norm_factor;
            tx_norm_factor = sqrt(mean(abs(tx_symbols_flat).^2));
            tx_symbols_flat = tx_symbols_flat / tx_norm_factor;
            
            Distorted_CDM = generate_CDM_Smooth(rx_symbols_flat, 64);
            Ideal_CDM     = generate_CDM_Smooth(tx_symbols_flat, 64);
            
            %% --- Step 8: 二维特征序列打包固化持久化存储 ---
            % 根据子文件夹编号分配湍流强度目录：文件夹1、4→weak，文件夹2、3→strong
            if sub_num == 1 || sub_num == 4
                save_dir = weak_dir;
            else
                save_dir = strong_dir;
            end
            % 使用全局帧编号global_t_pos(1~100)替代局部编号t(1~25)命名文件
            save_name = fullfile(save_dir, sprintf('exp_frame_%03d_mod_%d.mat', global_t_pos, Label_Bits));
            
            rx_IQ = [real(rx_symbols_flat), imag(rx_symbols_flat)];
            tx_IQ = [real(tx_symbols_flat), imag(tx_symbols_flat)];
            
            save(save_name, 'Distorted_CDM', 'Ideal_CDM', 'Label_Bits', ...
                'rx_symbols_flat', 'tx_symbols_flat', 'rx_IQ', 'tx_IQ');
            
            total_saved_count = total_saved_count + 1;
            fprintf('  [Exported] %s-Sub%02d File %02d -> 全局帧%03d CDM构建完成 (SNR=%.2f dB)\n', current_mod, sub_num, t, global_t_pos, avg_SNR);
        end
    end
end

%% 4. 全局统计报告输出
set(0, 'DefaultFigureVisible', 'on'); 
fprintf('\n==================================================================\n');
fprintf('  ✅ 实验信号帧特征数据集全局构建圆满成功\n');
fprintf('  累计采集有效留存样本：%d 份（过滤了深度信道衰落的失效帧）\n', total_saved_count);
fprintf('  实验CDM数据集存储路径：\n');
fprintf('  - 弱湍流（文件夹1、4）：%s\n', weak_dir);
fprintf('  - 强湍流（文件夹2、3）：%s\n', strong_dir);
fprintf('==================================================================\n');

%% ======================= 【附录辅助函数】=======================
function cdm = generate_CDM_Smooth(complex_symbols, grid_size)
    edges = linspace(-2.0, 2.0, grid_size + 1);
    [N, ~, ~] = histcounts2(real(complex_symbols), imag(complex_symbols), edges, edges);
    cdm = rot90(N);
    cdm = imgaussfilt(cdm, 1.0);
    if max(cdm(:)) > 0, cdm = cdm / max(cdm(:)); end
end