% snr_analysis.m
% 按照 snr分析.txt 对预处理结果做 SNR 统计，生成3张图
% 输入: dataset_lightprior/full_frame_16G/<mod>/<turb>/sig_XXXX_frameY.mat
% 参考: tx_3frame_6mod/<mod>/<sub>/sig_XXXX_frameY.mat (含 data_tx)
% 输出: dataset_lightprior/snr_analysis/{fig1,fig2,fig3}.png

clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));
global PROG; PROG.showMessagesLevel = 1; initProg();

%% ===================== 配置 =====================
data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
in_dir   = fullfile(data_root, 'dataset_lightprior', 'full_frame_16G');
tx_root  = fullfile(data_root, 'tx_3frame_6mod');
out_dir  = fullfile(data_root, 'dataset_lightprior', 'snr_analysis');
if ~exist(out_dir,'dir'), mkdir(out_dir); end

mod_names = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};
turb_names = {'weak', 'moderate', 'strong'};
turb_to_sub = containers.Map('KeyType','char','ValueType','char');
turb_to_sub('weak')    = 'sub1';
turb_to_sub('moderate')= 'sub2';
turb_to_sub('strong')  = 'sub3';

% OFDM 参数
ofdm.NumberOfIFFTSamples = 256;
ofdm.NumberOfGuardTime = 16;
ofdm.Carrier_location = 4:126;
ofdm.Carrier_location_demo = [4:126, 132:254];
ofdm.NumberOfCarriers = length(ofdm.Carrier_location);
ofdm.NumberOfCarriers_demo = length(ofdm.Carrier_location_demo);
ofdm.size = 128;
n_sc = ofdm.NumberOfCarriers;

% TX 参数模板
SIG.symRate = 8e9; SIG.nPol = 1; SIG.nSyms = 128;
SIG.rollOff = 0.25; SIG.modulation = 'QAM';

%% ===================== 逐调制处理 =====================
% all_snr{mod_idx, turb_idx} = [nFrames x nSc]
% all_fmean{mod_idx, turb_idx} = [nFrames x 1]

n_mod = length(mod_names);
n_turb = length(turb_names);

% 先统计帧数
frame_counts = zeros(n_mod, n_turb);
for m = 1:n_mod
    for t = 1:n_turb
        d = dir(fullfile(in_dir, mod_names{m}, turb_names{t}, '*.mat'));
        frame_counts(m, t) = length(d);
    end
end
max_fr = max(frame_counts(:));

all_snr = cell(n_mod, n_turb);
all_fmean = cell(n_mod, n_turb);
for m = 1:n_mod
    for t = 1:n_turb
        all_snr{m, t} = NaN(max_fr, n_sc);
        all_fmean{m, t} = NaN(max_fr, 1);
    end
end

% 关闭 deOFDM 内部的所有弹窗（星座图、帧头搜索等）
set(0, 'DefaultFigureVisible', 'off');

processed = 0;
total = sum(frame_counts(:));

for m = 1:n_mod
    mod_name = mod_names{m};
    % 调制阶数
    switch mod_name
        case 'QPSK',   Mq = 4;  bits = 2;
        case '16QAM',  Mq = 16; bits = 4;
        case '32QAM',  Mq = 32; bits = 5;
        case '64QAM',  Mq = 64; bits = 6;
        case '128QAM', Mq = 128;bits = 7;
        case '256QAM', Mq = 256;bits = 8;
    end
    nBpS_net = bits - 0.2*(bits > 2);
    
    % 初始化该调制的 TX.QAM
    TX.SIG = setSignalParams('symRate', SIG.symRate, 'M', Mq, ...
        'nPol', SIG.nPol, 'nBpS', nBpS_net, 'nSyms', SIG.nSyms, ...
        'roll-off', SIG.rollOff, 'modulation', SIG.modulation);
    TX.QAM = QAM_config(TX.SIG);
    C = TX.QAM.IQmap;
    DSP.DEMAPPER.normMethod = 'MMSE';
    DSP.DEMAPPER.normalizeTX = false;  % PS 成型下 data_tx 不覆盖全星座，跳过避免维度不兼容
    
    for t = 1:n_turb
        turb = turb_names{t};
        sub  = turb_to_sub(turb);
        files = dir(fullfile(in_dir, mod_name, turb, '*.mat'));
        if isempty(files), continue; end
        
        for f = 1:length(files)
            % 解析 sig_XXXX_frameY.mat
            tok = regexp(files(f).name, 'sig_(\d+)_frame(\d+)\.mat', 'tokens');
            if isempty(tok), continue; end
            sig_idx = str2double(tok{1}{1});
            fr_k    = str2double(tok{1}{2});
            
            %% 加载接收帧（16G时域信号）
            rx_file = fullfile(in_dir, mod_name, turb, files(f).name);
            tmp = load(rx_file);
            if isfield(tmp, 'sample')
                rx16 = double(tmp.sample.rx_frame_16_full);
            else
                continue;
            end
            
            %% 加载参考帧（data_tx = 频域符号）
            ref_file = fullfile(tx_root, mod_name, sub, ...
                sprintf('sig_%04d_frame%d.mat', sig_idx, fr_k));
            if ~exist(ref_file, 'file'), continue; end
            ref = load(ref_file);
            if isfield(ref, 'data_tx')
                tx_ref = ref.data_tx.';  % (nSyms, nCarriers) → (nCarriers, nSyms)
            else
                continue;
            end
            
            %% 手动解调（替换 deOFDM，避免帧头搜索边界问题）
            % 预处理帧已知是从帧头开始的，直接用LTS相关做细同步
            n_fft = ofdm.NumberOfIFFTSamples;
            n_guard = ofdm.NumberOfGuardTime;
            zh = 80;
            % 本地LTS时域参考
            load('LongTrainSym_ini.mat', 'LongTrainSym_ini');
            LTS_f = LongTrainSym_ini(1:n_fft);
            LTS_f([1 n_fft/2+1]) = 0;
            ltrs_in = LTS_f;
            ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));
            LTS_t = ifft(ltrs_in);
            rx16_row = rx16(:).';
            sym_len = n_fft + n_guard;
            % 一帧最小需求：帧头(80) + CP(16) + 2*LTS(512) + 至少1个OFDM符号(272)
            frm_min_len = zh + n_guard + 2*n_fft + sym_len;

            % 解调主体失败（LTS伪峰/帧长不足/NaN等）→ 该帧SNR置NaN，不中断批量
            rx_sc = [];
            try
                % 滑动相关找LTS
                xc = zeros(1, length(rx16_row)-n_fft+1);
                for ni = 1:length(xc)
                    xc(ni) = abs(sum(rx16_row(ni:ni+n_fft-1) .* conj(LTS_t)));
                end
                [~, lts_peak] = max(xc);
                % 帧起始 = LTS位置 - 80 - 16 - 1
                frm_start = max(1, lts_peak - zh - n_guard - 1);
                % 校验：从 frm_start 起必须能容纳最小帧体，否则跳过该帧
                if frm_start + frm_min_len - 1 > length(rx16_row)
                    error('frame too short from frm_start=%d (need>=%d, have=%d)', ...
                        frm_start, frm_min_len, length(rx16_row));
                end
                % 取完整一帧
                frm_end = min(length(rx16_row), frm_start + zh + n_guard + 2*n_fft + sym_len*SIG.nSyms - 1);
                rx_frm = rx16_row(frm_start:frm_end);
                % LTS提取
                lts1 = rx_frm(zh+n_guard+1 : zh+n_guard+n_fft);
                lts2 = rx_frm(zh+n_guard+n_fft+1 : zh+n_guard+2*n_fft);
                % CFO估计/校正
                pd = angle(sum(lts1.*conj(lts2)));
                cfo = pd/(2*pi*n_fft);
                rx_frm = rx_frm .* exp(-1j*2*pi*cfo*(0:length(rx_frm)-1)/n_fft);
                % 数据段
                ds = zh + n_guard + 2*n_fft + 1;
                dp = rx_frm(ds:end);
                nd = floor(length(dp)/sym_len);
                dp = dp(1:nd*sym_len);
                dm = reshape(dp, sym_len, nd);
                dn = dm(n_guard+1:end, :);
                fd = fft(dn, n_fft, 1)/sqrt(n_fft);
                % 信道估计
                lts_avg = (lts1(:)+lts2(:))/2;  % 列向量
                lts_fd = fft(lts_avg, n_fft)/sqrt(n_fft);
                H = lts_fd./(LTS_f(:)+1e-12);
                H(abs(LTS_f(:))<0.5) = 1;
                feq = fd./H;
                % 提取数据子载波
                carrier_loc = 4:126;
                rx_sc = feq(carrier_loc, :);  % (123, n_syms)
            catch ME
                fprintf('  [skip] %s %s sig%04d_frame%d: %s\n', ...
                    mod_name, turb, sig_idx, fr_k, ME.message);
                rx_sc = [];
            end
            
            %% symDemapper + EVM_eval
            % 解调失败或帧体过短的帧：跳过（all_snr/all_fmean 已预分配为 NaN）
            processed = processed + 1;
            if mod(processed, 100) == 0
                fprintf('Progress: %d/%d\n', processed, total);
            end
            if isempty(rx_sc)
                continue;
            end
            n_sym = min(size(rx_sc,2), size(tx_ref,2));
            rx_sc = rx_sc(:, 1:n_sym);
            tx_ref = tx_ref(:, 1:n_sym);
            
            txafdem = zeros(n_sc, n_sym);
            for sc = 1:n_sc
                try
                    DSP.DEMAPPER.N0 = 0;
                    [DSP.DEMAPPER, td] = symDemapper(rx_sc(sc,:), tx_ref(sc,:), C, DSP.DEMAPPER);
                    txafdem(sc,:) = td;
                catch ME
                    txafdem(sc,:) = NaN;  % 该子载波标记无效，后续 EVM/SNR 自动变 NaN 被过滤
                end
            end
            
            [~, SNR_sc] = EVM_eval(rx_sc, txafdem);
            all_snr{m, t}(f, :) = SNR_sc(:).';
            
            valid = SNR_sc(isfinite(SNR_sc) & SNR_sc > 0);
            if ~isempty(valid)
                all_fmean{m, t}(f) = 10*log10(mean(10.^(valid/10)));
            end
        end
    end
    fprintf('=== %s done (%d frames) ===\n', mod_name, sum(frame_counts(m,:)));
end
fprintf('\nTotal: %d/%d frames\n', processed, total);

%% ===================== 图1: 箱线图 =====================
figure('Position', [100 100 900 500]);
data_box = cell(1, n_mod);
for m = 1:n_mod
    v = [];
    for t = 1:n_turb
        v = [v; all_fmean{m, t}(isfinite(all_fmean{m, t}) & all_fmean{m, t} > 0)];
    end
    data_box{m} = v;
end
boxplot_group(data_box, mod_names);
ylabel('Frame-Level SNR (dB)');
title('SNR Distribution by Modulation Format');
grid on;
saveas(gcf, fullfile(out_dir, 'fig1_snr_boxplot.png'));
fprintf('Saved fig1\n');

%% ===================== 图2: 直方图 =====================
figure('Position', [100 100 1100 600]);
edges = 0:5:40;
colors = lines(n_mod);
for m = 1:n_mod
    v = [];
    for t = 1:n_turb
        v = [v; all_fmean{m, t}(isfinite(all_fmean{m, t}) & all_fmean{m, t} > 0)];
    end
    subplot(2,3,m);
    histogram(v, edges, 'FaceColor', colors(m,:), 'EdgeColor','w', 'Normalization','probability');
    xlabel('SNR (dB)'); ylabel('Prob'); title(mod_names{m});
    xlim([0 40]); grid on;
end
sgtitle('SNR Distribution per Modulation');
saveas(gcf, fullfile(out_dir, 'fig2_snr_histogram.png'));
fprintf('Saved fig2\n');

%% ===================== 图3: 平均子载波SNR =====================
figure('Position', [100 100 900 500]); hold on;
for m = 1:n_mod
    all_sc = [];
    for t = 1:n_turb
        all_sc = [all_sc; all_snr{m, t}];
    end
    mu = nanmean(all_sc, 1);
    plot(1:n_sc, mu, 'LineWidth', 1.5, 'DisplayName', mod_names{m});
end
hold off;
xlabel('Subcarrier'); ylabel('Avg SNR (dB)');
title('Average Per-Subcarrier SNR by Modulation');
legend; grid on; xlim([1 n_sc]);
saveas(gcf, fullfile(out_dir, 'fig3_subcarrier_snr.png'));
fprintf('Saved fig3\n');

%% 保存数据
save(fullfile(out_dir, 'snr_data.mat'), 'all_snr', 'all_fmean', ...
    'mod_names', 'turb_names', 'frame_counts', '-v7.3');
fprintf('\nAll saved to %s\nDone.\n', out_dir);

%% ===================== 箱线图函数 =====================
function boxplot_group(data, labels)
    n = length(data);
    hold on;
    colors = lines(n);
    for i = 1:n
        d = data{i}(:);
        d = d(isfinite(d));
        if length(d) < 5, continue; end
        q1 = prctile(d, 25); q2 = median(d); q3 = prctile(d, 75);
        iqr = q3 - q1;
        lo = max(min(d), q1 - 1.5*iqr);
        hi = min(max(d), q3 + 1.5*iqr);
        w = 0.5;
        patch([i-w i+w i+w i-w], [q1 q1 q3 q3], colors(i,:), ...
            'FaceAlpha', 0.25, 'EdgeColor', colors(i,:));
        plot([i-w i+w], [q2 q2], 'color', colors(i,:), 'LineWidth', 2);
        plot([i i], [lo hi], 'color', colors(i,:), 'LineWidth', 1);
        if any(d < lo), plot(i, lo, '^', 'color', colors(i,:)); end
        if any(d > hi), plot(i, hi, 'v', 'color', colors(i,:)); end
    end
    hold off;
    set(gca, 'XTick', 1:n, 'XTickLabel', labels);
end
