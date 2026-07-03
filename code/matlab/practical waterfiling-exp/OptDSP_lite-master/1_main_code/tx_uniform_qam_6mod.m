%% tx_uniform_qam_6mod.m — 均匀 QAM 发送信号生成（无 CCDM 概率整形）
% 基于 tx_1frame.m 修改，将 Tx_ProbShaping 替换为 Tx_QAM
% 6种调制格式：QPSK / 16QAM / 32QAM / 64QAM / 128QAM / 256QAM
% 第一阶段：10 bins × 3 frames = 30 frames per modulation

clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

global PROG;
PROG.showMessagesLevel = 2;
initProg();

RGB = fancyColors();
co = 1;

%% ===================== Basic signal config =====================

SIG.M = 4;
SIG.symRate = 8e9 / co;
SIG.bitRate_net = 8e9;
SIG.modulation = 'QAM';
SIG.rollOff = 0.25;
SIG.nPol = 1;
SIG.nSyms = 2^7 / co;   % 128 OFDM symbols per frame

nSpS = 5;
laserLW = 0e6;

FEC_rate = 1;
pilotRate = 1;
SNR_dB = 80;

%% ===================== OFDM config =====================

ofdm.NumberOfIFFTSamples = 256;

ofdm.Carrier_location = 4:126;
ofdm.Carrier_location_demo = [4:126, 132:254];

ofdm.NumberOfCarriers = length(ofdm.Carrier_location);
ofdm.NumberOfCarriers_demo = length(ofdm.Carrier_location_demo);

ofdm.NumberOfGuardTime = 16;

Fs = 10e9;
Fg = 10e9;

ofdm.size = SIG.nSyms;

%% ===================== TX config =====================

TX.BIT.source = 'randi';

TX.PS.type = 'RRC';
TX.PS.nTaps = 4096;

TX.LASER.linewidth = laserLW;

TX.PILOTS.active = true;
TX.PILOTS.rate = pilotRate;
TX.PILOTS.option = 'outerQPSK';

TX.FEC.active = false;
TX.FEC.rate = FEC_rate;
TX.FEC.nIter = 50;

% ⚠️ 注意：不使用 CCDM 概率整形，改用 Tx_QAM 均匀映射
% TX.PCS.method = 'CCDM';  ← 已移除

%% ===================== Modulation config =====================

% 6 种调制格式（均匀 QAM）
mod_bits  = [2, 4, 5, 6, 7, 8];
mod_names = {'QPSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM'};

% 第一阶段：10 bins × 3 frames = 30 frames per modulation
N_per_mod = 30;       % 每种调制格式生成 30 帧
N_per_sub = 3;        % 每个子文件夹放 3 帧
N_bins = N_per_mod / N_per_sub;  % = 10 bins

out_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\uniformQAM_6mod';

if ~exist(out_root, 'dir')
    mkdir(out_root);
end

%% ===================== Main loop =====================

for m = 1:length(mod_bits)

    bits = mod_bits(m);
    mname = mod_names{m};

    md = fullfile(out_root, mname);

    if ~exist(md, 'dir')
        mkdir(md);
    end

    SIG.M = 2^bits;

    % 均匀 QAM 不需要 CCDM 开销扣除
    nBpS_net = bits;

    TX.SIG = setSignalParams( ...
        'symRate', SIG.symRate, ...
        'M', SIG.M, ...
        'nPol', SIG.nPol, ...
        'nBpS', nBpS_net, ...
        'nSyms', SIG.nSyms, ...
        'roll-off', SIG.rollOff, ...
        'modulation', SIG.modulation);

    TX.QAM = QAM_config(TX.SIG);

    TX.BIT.seed = 100;

    TX.PS.rollOff = TX.SIG.rollOff;
    TX.DAC.RESAMP.sampRate = nSpS * TX.SIG.symRate;

    fprintf('\n============================================\n');
    fprintf('Generating %s (uniform QAM), M = %d, bits = %d\n', ...
        mname, SIG.M, bits);
    fprintf('============================================\n');

    for si = 1:N_per_mod

        sub_idx = floor((si - 1) / N_per_sub) + 1;
        sd = fullfile(md, sprintf('sub%02d', sub_idx));

        if ~exist(sd, 'dir')
            mkdir(sd);
        end

        % 每个信号使用独立随机种子
        TX.BIT.seed = 100 + m * 1000000 + si * 1000;

        %% ===================== Generate one OFDM frame =====================

        S.txofdm = [];

        for i = 1:ofdm.NumberOfCarriers

            TX.BIT.txBits1 = Tx_generateBits( ...
                SIG.nSyms, ...
                TX.QAM.M, ...
                TX.QAM.nPol, ...
                TX.BIT);

            % ⚠️ 关键修改：使用 Tx_QAM（均匀QAM）替代 Tx_ProbShaping（CCDM概率整形）
            [S.tx, txSyms] = Tx_QAM(TX.QAM, TX.BIT.txBits1);

            TX.BIT.seed = TX.BIT.seed + 1;

            % 每个有效子载波装载一组 QAM 符号
            S.txofdm(i, :) = 1 / sqrt(512) * S.tx;
        end

        % data_tx: [nSyms × nCarriers]
        S.txofdm = S.txofdm.';
        data_tx = S.txofdm;

        %% ===================== OFDM modulation =====================

        [S.txofdm] = OFDM(S.txofdm, ofdm, SIG.nSyms);

        % 16G -> 80G
        S.txSC = resample(S.txofdm, 80e9, 16e9);

        InputFSO = S.txSC.';

        %% ===================== 256-point alignment =====================

        alen = ceil(length(InputFSO) / 256) * 256;

        if length(InputFSO) < alen
            InputFSO = [InputFSO; zeros(alen - length(InputFSO), 1)];
        end

        %% ===================== Save =====================

        mat_file = fullfile(sd, sprintf('sig_%04d.mat', si));
        txt_file = fullfile(sd, sprintf('sig_%04d.txt', si));

        save(mat_file, 'data_tx');
        save(txt_file, 'InputFSO', '-ascii');

        if mod(si, 3) == 0 || si == N_per_mod
            fprintf('  %s: bin %02d frame %d/%d saved\n', mname, sub_idx, si, N_per_mod);
        end
    end
end

fprintf('\n============================================\n');
fprintf('All done! Uniform QAM (6 modulations) generated.\n');
fprintf('Output root:\n  %s\n', out_root);
fprintf('Per modulation: %d frames in %d bins\n', N_per_mod, N_bins);
fprintf('============================================\n');