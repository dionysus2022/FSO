% =========================================================================
% rx_exp_all_sub_maps.m - 5x4 级时频衰落热力图量产级物理层批处理脚本
% 功能：支持5种调制格式、每种格式4个子文件夹（共500帧文件）的全自动批处理
% 升级：分格式统计闪烁指数，并为全网 20 个子文件夹独立导出 2D 时频信道热力图
% =========================================================================
clear; clear global; close all; clc;
%% 1. 加载库与全局环境设置
addpath(genpath('D:\matlab\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));
global PROG; PROG.showMessagesLevel = 2; initProg(); RGB = fancyColors(); co=1;

% 核心防弹窗：默认关闭循环内部所有临时绘图窗口，仅在后台静默渲染并导出图片
set(0, 'DefaultFigureVisible', 'off');

%% 2. 5种调制格式+4个子文件夹循环边界定义
mod_list = {'2QAM', '4QAM', '16QAM', '64QAM', '256QAM'}; 
sub_list = {'1', '2', '3', '4'}; % 4个子文件夹
t_start = 1; 
t_end = 100;       % 每个子文件夹内固定存有25帧文件
scope_Fs = 80e9;   % 示波器原生硬件采样率

%% 3. 基础通信与OFDM参数设置（固定不变）
SIG.M = 4; SIG.symRate = 8e9/co; SIG.bitRate_net = 8e9;
SIG.modulation = 'QAM'; SIG.rollOff = 0.25; SIG.nPol = 1;
SIG.nSyms = 2^7/co; nSpS = 5; laserLW = 0e6;
FEC_rate = 1; pilotRate = 1; useCPE2 = false; SNR_dB = 80;
ofdm.NumberOfIFFTSamples=256; ofdm.Carrier_location=[4:126];
ofdm.Carrier_location_demo=[4:126,132:254];
ofdm.NumberOfCarriers=length(ofdm.Carrier_location);
ofdm.NumberOfCarriers_demo=length(ofdm.Carrier_location_demo);
ofdm.NumberOfGuardTime=16; Fs=10e9; Fg=10e9;
nBpS_net = SIG.bitRate_net/(SIG.nPol*SIG.symRate*FEC_rate*pilotRate);
TX.SIG = setSignalParams('symRate',SIG.symRate,'M',SIG.M,...
    'nPol',SIG.nPol,'nBpS',nBpS_net,'nSyms',SIG.nSyms,...
    'roll-off',SIG.rollOff,'modulation',SIG.modulation);
TX.QAM = QAM_config(TX.SIG); TX.BIT.source = 'randi'; TX.BIT.seed = 100;
TX.PS.type = 'RRC'; TX.PS.rollOff = TX.SIG.rollOff; TX.PS.nTaps = 4096;
TX.DAC.RESAMP.sampRate = nSpS*TX.SIG.symRate; TX.LASER.linewidth = laserLW;
TX.PILOTS.active = true; TX.PILOTS.rate = pilotRate; TX.PILOTS.option = 'outerQPSK';
TX.FEC.active = false; TX.FEC.rate = FEC_rate; TX.FEC.nIter = 50; TX.PCS.method = 'CCDM';
ofdm.size = SIG.nSyms;
C = TX.QAM.IQmap;
N_sc = ofdm.NumberOfCarriers;

% 基础DSP配置（固定不变）
DSP.MF.type='RRC'; DSP.MF.rollOff=TX.SIG.rollOff;
DSP.CPE1.method='pilot-based:optimized'; DSP.CPE1.decision='data-aided';
DSP.CPE1.nTaps_min=1; DSP.CPE1.nTaps_max=201; DSP.CPE1.PILOTS=TX.PILOTS;
DSP.CPE2.method='BPS'; DSP.CPE2.nTaps=22;
DSP.CPE2.nTaps_min=1; DSP.CPE2.nTaps_max=501;
DSP.CPE2.nTestPhases=10; DSP.CPE2.angleInterval=pi/8;
DSP.DEMAPPER.normMethod='MMSE';

%% 4. 全局根路径配置
data_root = 'D:\matlab\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';

%% ======================= 【第一层大循环：遍历各类调制格式】=======================
for m_idx = 1:length(mod_list)
    current_mod = mod_list{m_idx};
    fprintf('\n🚀 ==================== 开始处理调制格式 %s ====================\n', current_mod);
    
    % 初始化当前调制格式下全局总矩阵：123子载波 × 100时序帧（4个子文件夹×25帧）
    total_time_ticks = length(sub_list) * t_end;
    mod_global_snr = NaN(N_sc, total_time_ticks);
    mod_global_avg = NaN(total_time_ticks, 1);
    
    %% ======================= 【第二层大循环：遍历4个子文件夹】=======================
    for s_idx = 1:length(sub_list)
        current_sub = sub_list{s_idx};
        sub_num = str2double(current_sub);
        
        % 映射当前子文件夹相对绝对路径
        rx_dir  = fullfile(data_root, 'rx_data', current_mod, current_sub); 
        ref_dir = fullfile(data_root, 'tx_1frame_5mod', current_mod, sprintf('sub%02d', sub_num)); 
        
        % 自动匹配并创建最终数据保存路径
        save_dir = fullfile(data_root, 'results', current_mod, sprintf('sub%02d', sub_num));
        if ~exist(save_dir, 'dir'), mkdir(save_dir); end
        
        %% ======================= 【第三层内部循环：处理单个文件夹内25帧】=======================
        for t = t_start:t_end
            % 计算当前帧在1~100全局总时序上的绝对位置
            global_t_pos = (sub_num - 1) * t_end + t;
            
            rx_bin_file = fullfile(rx_dir, sprintf('%d.bin', t));
            ref_mat_file = fullfile(ref_dir, sprintf('sig_%04d.mat', t));
            
            % 多级安全校验：跳过缺失文件
            if ~exist(rx_bin_file, 'file'), continue; end
            if ~exist(ref_mat_file, 'file'), continue; end
            d = dir(rx_bin_file); if d.bytes == 0, continue; end
            
            %% [1] 精准二进制读取
            fid = fopen(rx_bin_file, 'rb');
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
                case 4, OutputFSO=fread(fid,num_points,'float32').';
                case 2, OutputFSO=fread(fid,num_points,'int16').';
                case 1, OutputFSO=fread(fid,num_points,'int8').';
                otherwise, OutputFSO=fread(fid,num_points,'double').';
            end
            fclose(fid);
            
            %% [2] 加载参考信号与前端重采样
            load(ref_mat_file, 'data_tx');
            OutputFSO=resample(OutputFSO,16e9,scope_Fs);
            data_in_mean2=mean(OutputFSO);
            OutputFSO=OutputFSO-data_in_mean2;
            data_in_Amp2=sum(abs(OutputFSO))/length(OutputFSO);
            AMP_rate2=1/data_in_Amp2;
            data_normal2=OutputFSO*AMP_rate2;
            data_in=data_normal2;
            
            %% [3] 原生 deOFDM 解调
            S.rx_1sps=data_in;
            S.rx_1sps=deOFDM(S.rx_1sps,ofdm,SIG.nSyms);
            if pilotRate<1, [S.rx_1sps,DSP.CPE1]=carrierPhaseEstimation(S.rx_1sps,S.tx,DSP.CPE1); end
            if useCPE2, [S.rx_1sps,DSP.CPE2]=carrierPhaseEstimation(S.rx_1sps,S.tx,DSP.CPE2,C); end
            if pilotRate<1, [S.rx_1sps,S.tx]=pilotSymbols_rmv(S.rx_1sps,S.tx,DSP.CPE1.PILOTS); end
            
            S.tx=data_tx.';
            S.rx_1sps=reshape(S.rx_1sps,SIG.nSyms,ofdm.NumberOfCarriers_demo);
            S.rx_1sps=S.rx_1sps.';
            
            %% [4] 判决与SNR计算
            S.BER = zeros(123, 1);
            S.txafdem_matrix = zeros(123, SIG.nSyms);
            for i=1:123
                [DSP.DEMAPPER,S.txafdem]=symDemapper(S.rx_1sps(i,:),S.tx(i,:),C,DSP.DEMAPPER);
                [BER,~]=BER_eval(DSP.DEMAPPER.txBits,DSP.DEMAPPER.rxBits);
                S.BER(i,:)=BER; DSP.DEMAPPER.N0=0;
                S.txafdem_matrix(i,:)=S.txafdem;
            end
            
            S.rx_1sps_eval=S.rx_1sps(1:123,:);
            [EVM,SNR_CAL]=EVM_eval(S.rx_1sps_eval,S.txafdem_matrix);
            
            v=SNR_CAL(SNR_CAL>0&isfinite(SNR_CAL));
            avg_SNR=10*log10(mean(10.^(v/10)));
            
            %% [5] 深衰落过滤清洗（阈值=3 dB）
            if avg_SNR < 3 || isnan(avg_SNR)
                fprintf('  [%s-Sub%s] 帧号 %d 遭遇深衰落(%.2f dB)，已清洗\n', current_mod, current_sub, t, avg_SNR);
                continue;
            end
            
            % 将当前有效帧数据写入全局大矩阵
            mod_global_snr(:, global_t_pos) = SNR_CAL;
            mod_global_avg(global_t_pos) = avg_SNR;
        end
        fprintf('  --> 格式 %s 下子文件夹 %s 处理完成\n', current_mod, current_sub);
        
        %% ======================= 【核心增改：分文件夹独立导出 2D 热力图 (共20张)】 =======================
        % 提取当前子文件夹映射的 25 帧局域频域数据
        time_block_indices = (sub_num - 1) * t_end + 1 : sub_num * t_end;
        sub_data = mod_global_snr(:, time_block_indices);
        
        % 检查该子文件夹内是否有存活的干净数据列
        valid_cols = any(~isnan(sub_data), 1);
        if any(valid_cols)
            fh_sub = figure('Visible', 'off', 'Color', 'w');
            % 绘图时横轴映射为局部文件的真实帧序号（1 ~ 25）
            imagesc(find(valid_cols), 1:N_sc, sub_data(:, valid_cols));
            colorbar; caxis([0 25]); colormap(jet);
            set(gca, 'YDir', 'normal');
            grid on;
            xlabel('局部时序帧号 (File 1~25)', 'FontSize', 10, 'FontWeight', 'bold');
            ylabel('OFDM 子载波序号 (1 ~ 123)', 'FontSize', 10, 'FontWeight', 'bold');
            title(sprintf('%s - 子文件夹 sub%02d SNR 信道分布图', current_mod, sub_num), 'FontSize', 11, 'FontWeight', 'bold');
            
            % 命名学术化：如 2QAM_sub01_2d_fading_map.png
            img_name = sprintf('%s_sub%02d_2d_fading_map.png', current_mod, sub_num);
            saveas(fh_sub, fullfile(save_dir, img_name));
            close(fh_sub);
            fprintf('    [Saved Sub-Map] 已成功导出局部热力图: %s\n', img_name);
        else
            fprintf('    [Warning] 子文件夹 sub%02d 内有效数据不足，未生成局部热力图。\n', sub_num);
        end
        % =========================================================================================
        
    end
    
    %% ======================= 【当前调制格式大结果汇总与学术绘图】=======================
    set(0, 'DefaultFigureVisible', 'on'); % 开启绘图状态用于最终全局图呈现
    
    valid_frames = mod_global_avg(~isnan(mod_global_avg));
    
    fprintf('\n📊 ========== 格式 %s 的全局大气信道分析报告 ==========\n', current_mod);
    if length(valid_frames) >= 2
        fprintf('  留存有效总帧数：%d / 100\n', length(valid_frames));
        fprintf('  干净样本平均SNR: %.2f dB\n', mean(valid_frames));
        fprintf('  对数信噪比方差: %.4f (dB^2)\n', var(valid_frames));
        
        % 线性光强物理闪烁指数计算
        linear_int = 10.^(valid_frames / 10);
        scint_index = var(linear_int) / (mean(linear_int)^2);
        fprintf('  全局综合闪烁指数: %.4f\n', scint_index);
        
        %% 绘制 100 帧全局长周期静态时频二维衰落热力图（每种格式独立一张，共5张）
        valid_cols = any(~isnan(mod_global_snr), 1);
        if any(valid_cols)
            fh_global = figure('Name', [current_mod, ' 全局信道衰落热力图'], 'Color', 'w');
            imagesc(find(valid_cols), 1:N_sc, mod_global_snr(:, valid_cols));
            colorbar; caxis([0 25]); colormap(jet);
            set(gca, 'YDir', 'normal');
            xlabel('实验总组数 / 时序轴 (全局 1~100 帧)', 'FontSize', 11, 'FontWeight', 'bold');
            ylabel('OFDM 子载波序号 (1 ~ 123)', 'FontSize', 11, 'FontWeight', 'bold');
            title(sprintf('%s 接收信号下的全局 FSO 子载波时域多径衰落热力图', current_mod), 'FontSize', 12, 'FontWeight', 'bold');
            grid on;
            
            % 将大图与总体干净特征矩阵保存在格式根目录下
            mod_root_save = fullfile(data_root, 'results', current_mod);
            if ~exist(mod_root_save, 'dir'), mkdir(mod_root_save); end
            saveas(fh_global, fullfile(mod_root_save, [current_mod, '_global_100frames_map.png']));
            save(fullfile(mod_root_save, [current_mod, '_all_100frames_cleaned.mat']), 'mod_global_snr', 'mod_global_avg');
            fprintf('  [Saved Global] 100帧全局总热力图及干净特征矩阵已存至：%s\n', mod_root_save);
        end
    else
        fprintf('  [Warning] 当前调制格式下留存帧数过少，跳过全局绘图\n');
    end
    
    set(0, 'DefaultFigureVisible', 'off'); % 重新关闭窗口可见性，静默进入下一种格式处理
end

set(0, 'DefaultFigureVisible', 'on'); 
fprintf('\n==================================================================\n');
fprintf('✅ 任务圆满结束！全网 20 个子文件夹的局部 2D 热力图与 5 张全局总图全部离线导出完毕。\n');