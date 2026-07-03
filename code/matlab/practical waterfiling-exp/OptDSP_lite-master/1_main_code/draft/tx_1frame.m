%% tx_1frame_6mod.m - single frame, 6 modulation formats
% QPSK / 16QAM / 32QAM / 64QAM / 128QAM / 256QAM
% 每个 sig_xxxx.txt 对应 1 帧发送波形
% 每个 sig_xxxx.mat 保存对应 data_tx 参考符号

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

% 注意：这里是每帧 OFDM symbols 数
% 你原始代码是 2^9 = 512，保持不变
SIG.nSyms = 2^7 / co;

nSpS = 5;
laserLW = 0e6;

FEC_rate = 1;
pilotRate = 1;
useCPE2 = false;
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

TX.PCS.method = 'CCDM';

%% ===================== Modulation config =====================

% 原代码是 2QAM / 4QAM / 16QAM / 64QAM / 256QAM
% 这里改为 6 类：
% QPSK / 16QAM / 32QAM / 64QAM / 128QAM / 256QAM

mod_bits  = [2, 4, 5, 6, 7, 8];
mod_names = {'QPSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM'};

% 每种调制格式生成多少个一帧信号
% 要生成 200 个就改成 200；要生成 500 个就保持 500
N_per_mod = 200;

% 每个子文件夹放多少个信号
% 例如 N_per_mod=500, N_per_sub=100，则生成 sub01~sub05
N_per_sub = 25;

out_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\tx_1frame_6mod_128sym';

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

    % =============================================================
    % CCDM / PCS 预留开销
    % 高阶 QAM 减去 0.2 bit，避免高阶星座符号分布异常
    % QPSK 不减
    % =============================================================
    if bits <= 2
        nBpS_net = bits;
    else
        nBpS_net = bits - 0.2;
    end

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
    fprintf('Generating %s, M = %d, bits = %d, nBpS_net = %.2f\n', ...
        mname, SIG.M, bits, nBpS_net);
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

            [S.tx, txSyms, TX.QAM] = Tx_ProbShaping( ...
                TX.BIT.txBits1, ...
                TX.QAM, ...
                TX.SIG, ...
                TX.FEC.rate);

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

        if mod(si, 10) == 0 || si == N_per_mod
            fprintf('  %s: %d / %d saved\n', mname, si, N_per_mod);
        end
    end
end

fprintf('\nAll done!\n');
fprintf('Output root:\n%s\n', out_root);