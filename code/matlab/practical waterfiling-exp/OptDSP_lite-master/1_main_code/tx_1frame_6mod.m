% tx_1frame_6mod.m - single frame，生成QPSK 16QAM 32QAM 64QAM 128QAM 256QAM信号
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
TX.FEC.nIter = 50; TX.PCS.method = 'CCDM';

%% Modulation config - QPSK, 16QAM, 32QAM, 64QAM, 128QAM, 256QAM
mod_bits = [2, 4, 5, 6, 7, 8];
mod_names = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};
N_per_mod = 200;   % 快速测试：每种调制生成1个信号
N_per_sub = 25;   % 每个子文件夹放1个信号
N_sub = 8;       % 1个子文件夹

out_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\tx_1frame_6mod0607';
if ~exist(out_root,'dir'), mkdir(out_root); end

for m = 1:length(mod_bits)
    bits = mod_bits(m); mname = mod_names{m};
    md = fullfile(out_root, mname);
    if ~exist(md,'dir'), mkdir(md); end

    SIG.M = 2^bits;
    nBpS_net = bits;        % 默认：每符号比特数跟随调制阶数
    % =============================================================
    % 【修复点】预留 CCDM 匹配开销，防止高阶调制的"符号位饿死"导致半边星座图
    % =============================================================
    if bits <= 2
        nBpS_net = bits;        % QPSK 没有幅度变化，直接拉满
    else
        nBpS_net = bits - 0.2;  % 16/32/64/128/256QAM 减去 0.2 bit 开销，留出正负号余量
    end
    TX.SIG = setSignalParams('symRate',SIG.symRate,'M',SIG.M,...
        'nPol',SIG.nPol,'nBpS',nBpS_net,'nSyms',SIG.nSyms,...
        'roll-off',SIG.rollOff,'modulation',SIG.modulation);
    TX.QAM = QAM_config(TX.SIG); TX.BIT.seed = 100;
    TX.PS.rollOff = TX.SIG.rollOff; TX.DAC.RESAMP.sampRate = nSpS*TX.SIG.symRate;

    fprintf('\n--- %s ---\n', mname);

    for si = 1:N_per_mod
        sub_idx = floor((si-1)/N_per_sub) + 1;  % sub01~sub10
        sd = fullfile(md, sprintf('sub%02d', sub_idx));
        if ~exist(sd,'dir'), mkdir(sd); end

        TX.BIT.seed = 100 + m*1000000 + si*1000;

        % Generate signal (same as untitled.m)
        S.txofdm = [];
        for i = 1:ofdm.NumberOfCarriers
            TX.BIT.txBits1 = Tx_generateBits(SIG.nSyms,TX.QAM.M,TX.QAM.nPol,TX.BIT);
            [S.tx, txSyms, TX.QAM] = Tx_ProbShaping(TX.BIT.txBits1, TX.QAM, TX.SIG, TX.FEC.rate);
            TX.BIT.seed = TX.BIT.seed + 1;
            S.txofdm(i,:) = 1/sqrt(512)*S.tx;
        end

        S.txofdm = S.txofdm.'; data_tx = S.txofdm;
        [S.txofdm] = OFDM(S.txofdm, ofdm, SIG.nSyms);
        S.txSC = resample(S.txofdm, 80e9, 16e9);
        InputFSO = S.txSC.';

        % 256-point alignment
        alen = ceil(length(InputFSO)/256)*256;
        if length(InputFSO) < alen
            InputFSO = [InputFSO; zeros(alen-length(InputFSO),1)];
        end

        % Save
        save(fullfile(sd, sprintf('sig_%04d.mat', si)), 'data_tx');
        save(fullfile(sd, sprintf('sig_%04d.txt', si)), 'InputFSO', '-ascii');

        if mod(si, 10)==0, fprintf('  %d/%d\n', si, N_per_mod); end
    end
end
fprintf('\nAll done!\n');