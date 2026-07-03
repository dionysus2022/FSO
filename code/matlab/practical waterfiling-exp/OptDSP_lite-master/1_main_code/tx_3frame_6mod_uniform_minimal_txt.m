% tx_3frame_6mod_uniform_minimal_txt.m
% 基于原 tx_3frame_6mod.m 的“最小改动版”
% 目的：只把 CCDM/Tx_ProbShaping 概率整形信号改成“均匀 QAM”信号
% 保持：原 OFDM、重采样、256点补齐、3帧拼接、.txt 输出、目录结构基本不变
%
% 6种调制：QPSK(4QAM)、16QAM、32QAM、64QAM、128QAM、256QAM
% 默认：每种100个信号，分4组(sub1~sub4)，每组25个
%
% 如果你要每种调制50个txt、2个子文件夹，只需要改：
%   N_per_mod = 50;
%   N_per_sub = 25;
%   N_sub = 2;

clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

global PROG; PROG.showMessagesLevel = 2; initProg(); RGB = fancyColors(); co = 1;
SIG.M = 4; SIG.symRate = 8e9/co; SIG.bitRate_net = 8e9; SIG.modulation = 'QAM';

SIG.rollOff = 0.25; SIG.nPol = 1; SIG.nSyms = 2^7/co; nSpS = 5; laserLW = 0e6;

FEC_rate = 1; pilotRate = 1; useCPE2 = false; SNR_dB = 80;

ofdm.NumberOfIFFTSamples=256; ofdm.Carrier_location=[4:126];

ofdm.Carrier_location_demo=[4:126,132:254];

ofdm.NumberOfCarriers=length(ofdm.Carrier_location);

ofdm.NumberOfCarriers_demo=length(ofdm.Carrier_location_demo);

ofdm.NumberOfGuardTime=16; Fs=10e9; Fg=10e9; ofdm.size = SIG.nSyms;

TX.BIT.source = 'randi'; TX.PS.type = 'RRC'; TX.PS.nTaps = 4096;

TX.LASER.linewidth = laserLW; TX.PILOTS.active = true; TX.PILOTS.rate = pilotRate;

TX.PILOTS.option = 'outerQPSK'; TX.FEC.active = false; TX.FEC.rate = FEC_rate;

TX.FEC.nIter = 50;

% =============================================================
% 【最小改动点 1】关闭 PCS/CCDM 概率整形标记
% 说明：后面不再调用 Tx_ProbShaping，本变量只作为记录，避免误解。
% =============================================================
TX.PCS.method = 'none';

%% Modulation config — 6 modulations
mod_bits = [2, 4, 5, 6, 7, 8];        % QPSK,16,32,64,128,256QAM
mod_names = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};

N_per_mod = 100;   % 每种调制生成100个信号
N_per_sub = 25;    % 每个子文件夹放25个
N_sub = 4;         % 4个子文件夹 sub1~sub4

out_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\tx_3frame_6mod_uniform_txt1';

if ~exist(out_root,'dir'), mkdir(out_root); end

for m = 1:length(mod_bits)
    bits = mod_bits(m); mname = mod_names{m};
    md = fullfile(out_root, mname);
    if ~exist(md,'dir'), mkdir(md); end

    SIG.M = 2^bits;

    % =============================================================
    % 【最小改动点 2】均匀QAM不需要 bits-0.2 的 CCDM 开销
    % 原概率整形版本中高阶QAM用 bits-0.2；均匀QAM必须使用完整bits。
    % =============================================================
    nBpS_net = bits;

    TX.SIG = setSignalParams('symRate',SIG.symRate,'M',SIG.M,...
        'nPol',SIG.nPol,'nBpS',nBpS_net,'nSyms',SIG.nSyms,...
        'roll-off',SIG.rollOff,'modulation',SIG.modulation);

    TX.QAM = QAM_config(TX.SIG); TX.BIT.seed = 100;
    TX.PS.rollOff = TX.SIG.rollOff; TX.DAC.RESAMP.sampRate = nSpS*TX.SIG.symRate;

    % =============================================================
    % 【最小改动点 3】直接使用工具箱 QAM_config 生成的星座点集合
    % 这样 32QAM/128QAM 仍跟你原系统定义保持一致。
    % =============================================================
    qam_const = TX.QAM.IQmap(:).';
    qam_const = qam_const(~isnan(real(qam_const)) & ~isnan(imag(qam_const)));
    qam_const = qam_const(isfinite(real(qam_const)) & isfinite(imag(qam_const)));

    if numel(qam_const) ~= SIG.M
        warning('%s: QAM_config returned %d constellation points, expected M=%d. Please check QAM_config/IQmap.', ...
            mname, numel(qam_const), SIG.M);
    end

    fprintf('\n--- %s | Uniform QAM | M=%d | constellation points=%d ---\n', ...
        mname, SIG.M, numel(qam_const));

    for si = 1:N_per_mod
        sub_idx = floor((si-1)/N_per_sub) + 1;  % sub1~sub4
        sd = fullfile(md, sprintf('sub%d', sub_idx));
        if ~exist(sd,'dir'), mkdir(sd); end

        TX.BIT.seed = 100 + m*1000000 + si*1000;

        % 初始化3帧拼接的存储变量
        data_tx_all = [];
        InputFSO_all = [];

        % 循环生成3帧信号并拼接（与原代码一致）
        for frame_idx = 1:3
            % Generate uniform-QAM OFDM frequency-domain signal
            S.txofdm = [];

            for i = 1:ofdm.NumberOfCarriers
                % =============================================================
                % 【最小改动点 4：核心】
                % 原代码：Tx_generateBits + Tx_ProbShaping(CCDM)
                % 新代码：直接等概率随机抽取星座点，生成均匀QAM符号
                % =============================================================
                rng(TX.BIT.seed, 'twister');
                sym_idx = randi(numel(qam_const), 1, SIG.nSyms);
                S.tx = qam_const(sym_idx);
                TX.BIT.seed = TX.BIT.seed + 1;

                % 保持原来的子载波缩放方式不变
                S.txofdm(i,:) = 1/sqrt(512)*S.tx;
            end

            % 后续流程完全保持原代码
            S.txofdm = S.txofdm.'; data_tx = S.txofdm;
            [S.txofdm] = OFDM(S.txofdm, ofdm, SIG.nSyms);
            S.txSC = resample(S.txofdm, 80e9, 16e9);
            InputFSO = S.txSC.';

            % 256-point alignment
            alen = ceil(length(InputFSO)/256)*256;
            if length(InputFSO) < alen
                InputFSO = [InputFSO; zeros(alen-length(InputFSO),1)];
            end

            % 拼接当前帧到总信号
            data_tx_all = [data_tx_all; data_tx];
            InputFSO_all = [InputFSO_all; InputFSO];
        end

        % Save 拼接后的3帧信号（合并文件）
        save(fullfile(sd, sprintf('sig_%04d.mat', si)), 'data_tx_all');

        % 输出仍然保持 .txt，兼容原 AWG 加载方式
        save(fullfile(sd, sprintf('sig_%04d.txt', si)), 'InputFSO_all', '-ascii');

        % 每帧另存为独立 .mat 文件，供解调时逐帧加载参考
        for fi = 1:3
            data_tx = data_tx_all((fi-1)*SIG.nSyms+1 : fi*SIG.nSyms, :);
            save(fullfile(sd, sprintf('sig_%04d_frame%d.mat', si, fi)), 'data_tx');
        end

        if mod(si, 10)==0, fprintf('  %d/%d\n', si, N_per_mod); end
    end
end

fprintf('\nAll done! Uniform QAM txt signals saved to:\n%s\n', out_root);
