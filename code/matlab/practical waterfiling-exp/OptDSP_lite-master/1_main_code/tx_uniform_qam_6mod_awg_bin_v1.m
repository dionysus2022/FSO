% tx_uniform_qam_6mod_awg_bin_v1.m
% ------------------------------------------------------------
% 基于你原来 AWG 能发送的 tx_3frame_6mod.m 版本最小修改：
% 1) 保留原 OFDM、resample、256 点补齐、InputFSO_all 拼接流程；
% 2) 去掉 Tx_ProbShaping / CCDM，改为本地均匀 QAM 星座采样；
% 3) 每个文件包含 3 帧；
% 4) 每种调制生成 50 个 .bin 文件，分 2 个子文件夹 sub1/sub2，每个子文件夹 25 个；
% 5) .bin 保存格式：float32 interleaved I/Q，即 real1 imag1 real2 imag2 ...
%
% 注意：如果你的 AWG 只接受 .txt，可将 SAVE_TXT_COPY=true，同时保留原来的 ASCII 输出。
% ------------------------------------------------------------

clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

global PROG; PROG.showMessagesLevel = 2; initProg(); RGB = fancyColors(); co = 1;

%% ===================== 基础参数：保持原可发送版本 =====================
SIG.M = 4;
SIG.symRate = 8e9/co;
SIG.bitRate_net = 8e9;
SIG.modulation = 'QAM';
SIG.rollOff = 0.25;
SIG.nPol = 1;
SIG.nSyms = 2^7/co;      % 128 OFDM symbols per frame
nSpS = 5;
laserLW = 0e6;

FEC_rate = 1;
pilotRate = 1;
useCPE2 = false; %#ok<NASGU>
SNR_dB = 80;     %#ok<NASGU>

ofdm.NumberOfIFFTSamples = 256;
ofdm.Carrier_location = [4:126];
ofdm.Carrier_location_demo = [4:126,132:254];
ofdm.NumberOfCarriers = length(ofdm.Carrier_location);          % 123
ofdm.NumberOfCarriers_demo = length(ofdm.Carrier_location_demo);
ofdm.NumberOfGuardTime = 16;
Fs = 10e9; %#ok<NASGU>
Fg = 10e9; %#ok<NASGU>
ofdm.size = SIG.nSyms;

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
TX.PCS.method = 'none_uniform_qam';   % 关键：不再使用 CCDM / PCS

%% ===================== 批量生成配置 =====================
mod_bits  = [2, 4, 5, 6, 7, 8];
mod_names = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};

N_per_mod = 50;       % 每种调制 50 个 bin
N_sub     = 2;        % 2 个子文件夹
N_per_sub = 25;       % 每个子文件夹 25 个 bin
N_frames_per_bin = 3; % 每个 bin 拼接 3 帧

SAVE_TXT_COPY = false;  % 若 AWG 仍需 txt，将这里改成 true
SAVE_INPUTFSO_MAT = true;

out_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\tx_3frame_6mod_uniform_bin';
if ~exist(out_root,'dir'), mkdir(out_root); end

fprintf('\n============================================================\n');
fprintf('Uniform QAM TX generation for AWG-compatible waveform\n');
fprintf('Output root: %s\n', out_root);
fprintf('Each modulation: %d bin files, %d frames/bin, %d subfolders\n', N_per_mod, N_frames_per_bin, N_sub);
fprintf('BIN format: float32 interleaved I/Q\n');
fprintf('============================================================\n');

%% ===================== 主循环 =====================
for m = 1:length(mod_bits)
    bits = mod_bits(m);
    mname = mod_names{m};
    md = fullfile(out_root, mname);
    if ~exist(md,'dir'), mkdir(md); end

    SIG.M = 2^bits;
    nBpS_net = bits;  % 均匀 QAM 不需要 CCDM shaping 开销

    TX.SIG = setSignalParams('symRate',SIG.symRate,'M',SIG.M,...
        'nPol',SIG.nPol,'nBpS',nBpS_net,'nSyms',SIG.nSyms,...
        'roll-off',SIG.rollOff,'modulation',SIG.modulation);

    % 保留 QAM_config，保证后续系统变量结构不破坏；但符号实际由 local constellation 生成
    TX.QAM = QAM_config(TX.SIG);
    TX.BIT.seed = 100;
    TX.PS.rollOff = TX.SIG.rollOff;
    TX.DAC.RESAMP.sampRate = nSpS*TX.SIG.symRate;

    % 本地均匀星座，平均功率归一化到 1
    const = local_uniform_qam_constellation(mname);
    const = const(:).';
    const = const ./ sqrt(mean(abs(const).^2));

    fprintf('\n--- %s, M=%d, constellation points=%d ---\n', mname, SIG.M, numel(const));
    fprintf('Constellation average power after normalization: %.6f\n', mean(abs(const).^2));

    for si = 1:N_per_mod
        sub_idx = floor((si-1)/N_per_sub) + 1;  % sub1/sub2
        if sub_idx > N_sub
            error('sub_idx exceeds N_sub. Check N_per_mod, N_per_sub, N_sub.');
        end
        sd = fullfile(md, sprintf('sub%d', sub_idx));
        if ~exist(sd,'dir'), mkdir(sd); end

        base_seed = 100 + m*1000000 + si*1000;
        rng(base_seed, 'twister');

        data_tx_all = [];
        InputFSO_all = [];

        for frame_idx = 1:N_frames_per_bin
            S.txofdm = [];

            % 每个有效子载波独立均匀采样 QAM 星座点
            for carrier_idx = 1:ofdm.NumberOfCarriers
                idx = randi(numel(const), 1, SIG.nSyms);
                S.tx = const(idx);

                % 保持原版每子载波缩放，避免改变 AWG 幅度尺度
                S.txofdm(carrier_idx,:) = 1/sqrt(512) * S.tx;
            end

            % 频域 OFDM 数据，大小：128 × 123
            S.txofdm = S.txofdm.';
            data_tx = S.txofdm;

            % 保持原版 OFDM + 重采样 + 256 点对齐
            S.txofdm = OFDM(S.txofdm, ofdm, SIG.nSyms);
            S.txSC = resample(S.txofdm, 80e9, 16e9);
            InputFSO = S.txSC.';

            alen = ceil(length(InputFSO)/256)*256;
            if length(InputFSO) < alen
                InputFSO = [InputFSO; zeros(alen-length(InputFSO),1)];
            end

            data_tx_all = [data_tx_all; data_tx];          %#ok<AGROW>
            InputFSO_all = [InputFSO_all; InputFSO];       %#ok<AGROW>
        end

        %% ========== 保存参考 MAT ==========
        save(fullfile(sd, sprintf('sig_%04d.mat', si)), ...
            'data_tx_all', 'InputFSO_all', 'mname', 'bits', 'SIG', 'ofdm', 'const', '-v7.3');

        for fi = 1:N_frames_per_bin
            row_idx = (fi-1)*SIG.nSyms + (1:SIG.nSyms);
            data_tx = data_tx_all(row_idx, :); %#ok<NASGU>
            save(fullfile(sd, sprintf('sig_%04d_frame%d.mat', si, fi)), 'data_tx', 'mname', 'bits', 'const');
        end

        %% ========== 保存 AWG BIN: float32 interleaved I/Q ==========
        bin_name = fullfile(sd, sprintf('sig_%04d.bin', si));
        write_complex_float32_iq(bin_name, InputFSO_all);

        %% ========== 可选保存 TXT 副本 ==========
        if SAVE_TXT_COPY
            save(fullfile(sd, sprintf('sig_%04d.txt', si)), 'InputFSO_all', '-ascii');
        end

        if SAVE_INPUTFSO_MAT
            save(fullfile(sd, sprintf('sig_%04d_waveform.mat', si)), 'InputFSO_all', '-v7.3');
        end

        if mod(si, 10)==0 || si==1
            fprintf('  saved %s: %d/%d\n', fullfile(sd, sprintf('sig_%04d.bin', si)), si, N_per_mod);
        end
    end
end

fprintf('\nAll uniform QAM bin files generated successfully.\n');

%% =====================================================================
function c = local_uniform_qam_constellation(mod_name)
% 生成均匀 QAM 星座点集合。点集合不含概率整形；采样时所有点等概率。
    switch upper(mod_name)
        case 'QPSK'
            c = [1+1j, 1-1j, -1+1j, -1-1j];

        case '16QAM'
            lv = [-3 -1 1 3];
            [I,Q] = meshgrid(lv, lv);
            c = I(:) + 1j*Q(:);

        case '32QAM'
            % 6×6 meshgrid 去四角，标准 32-cross QAM
            lv = [-5 -3 -1 1 3 5];
            [I,Q] = meshgrid(lv, lv);
            mask = ~((abs(I)==5) & (abs(Q)==5));
            c = I(mask) + 1j*Q(mask);

        case '64QAM'
            lv = [-7 -5 -3 -1 1 3 5 7];
            [I,Q] = meshgrid(lv, lv);
            c = I(:) + 1j*Q(:);

        case '128QAM'
            % 12×12 meshgrid 去四个 2×2 角块，标准 128-cross QAM
            lv = [-11 -9 -7 -5 -3 -1 1 3 5 7 9 11];
            [I,Q] = meshgrid(lv, lv);
            mask = ~((abs(I)>=9) & (abs(Q)>=9));
            c = I(mask) + 1j*Q(mask);

        case '256QAM'
            lv = [-15 -13 -11 -9 -7 -5 -3 -1 1 3 5 7 9 11 13 15];
            [I,Q] = meshgrid(lv, lv);
            c = I(:) + 1j*Q(:);

        otherwise
            error('Unsupported modulation: %s', mod_name);
    end
end

%% =====================================================================
function write_complex_float32_iq(filename, x)
% 将复数列向量保存为 float32 interleaved I/Q: real1 imag1 real2 imag2 ...
    x = x(:);
    iq = zeros(2*numel(x), 1, 'single');
    iq(1:2:end) = single(real(x));
    iq(2:2:end) = single(imag(x));

    fid = fopen(filename, 'wb');
    if fid < 0
        error('Cannot open file for writing: %s', filename);
    end
    fwrite(fid, iq, 'single');
    fclose(fid);
end
