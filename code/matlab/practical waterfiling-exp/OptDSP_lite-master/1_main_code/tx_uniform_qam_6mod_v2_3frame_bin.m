%% tx_uniform_qam_6mod_v2_3frame_bin.m
% 均匀 QAM 发送信号生成：无 CCDM / 无 PCS / 每个 bin 含 3 帧
%
% 输出结构：
%   uniformQAM_6mod_tx_v2/
%       QPSK/bin_0001/
%           sig_0001.bin          % 3帧拼接，float32 interleaved I/Q
%           sig_0001.txt          % 3帧拼接，两列: real imag
%           sig_0001.mat          % 3帧拼接 + 每帧 data_tx
%           frame_0001.mat        % 第1帧 TX 频域符号 + 时域波形
%           frame_0002.mat
%           frame_0003.mat
%
% 重要说明：
%   1) 这里使用 Tx_QAM()，不是 Tx_ProbShaping()，因此是均匀 QAM。
%   2) data_tx 中每个子载波的 QAM 符号乘了 1/sqrt(512)，与原 OFDM 链路保持一致。
%   3) 验证脚本也必须使用同样的 1/sqrt(512) 缩放理想星座。
%   4) 如果你的 AWG/FSO 发射端只接受 txt，就使用 sig_xxxx.txt；
%      如果接受二进制 float32 interleaved I/Q，就使用 sig_xxxx.bin。

clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

global PROG;
PROG.showMessagesLevel = 2;
initProg();

%% ===================== Basic signal config =====================

co = 1;
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

%% ===================== OFDM config =====================

ofdm.NumberOfIFFTSamples = 256;
ofdm.Carrier_location = 4:126;
ofdm.Carrier_location_demo = [4:126, 132:254];
ofdm.NumberOfCarriers = length(ofdm.Carrier_location);
ofdm.NumberOfCarriers_demo = length(ofdm.Carrier_location_demo);
ofdm.NumberOfGuardTime = 16;
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

% 不使用 CCDM / PCS：
% TX.PCS.method = 'CCDM';    % 禁用
% Tx_ProbShaping(...)        % 禁用
% 本脚本使用 Tx_QAM(...)

%% ===================== Modulation and output config =====================

mod_bits  = [2, 4, 5, 6, 7, 8];
mod_names = {'QPSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM'};

% 小规模验证阶段：每种调制 10 个 bin，每个 bin 3 帧，共 30 帧
N_bins_per_mod = 10;
N_frames_per_bin = 3;

% 如果接收端 packet detection 容易粘连，可改为 1024 或 2048。
% 先保持 0，尽量与原始 3-frame 连续发送逻辑一致。
gap_len_samples = 0;

carrier_scale = 1 / sqrt(512);

out_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\uniformQAM_6mod_tx_v2';

if ~exist(out_root, 'dir')
    mkdir(out_root);
end

%% ===================== Main loop =====================

for m = 1:length(mod_bits)

    bits = mod_bits(m);
    mname = mod_names{m};
    Mq = 2^bits;

    md = fullfile(out_root, mname);
    if ~exist(md, 'dir')
        mkdir(md);
    end

    SIG.M = Mq;

    % 均匀 QAM：不扣除 CCDM shaping overhead
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

    TX.PS.rollOff = TX.SIG.rollOff;
    TX.DAC.RESAMP.sampRate = nSpS * TX.SIG.symRate;

    fprintf('\n============================================\n');
    fprintf('Generating %s uniform QAM, M=%d, bits=%d\n', mname, Mq, bits);
    fprintf('Output: %s\n', md);
    fprintf('============================================\n');

    for bin_idx = 1:N_bins_per_mod

        bin_dir = fullfile(md, sprintf('bin_%04d', bin_idx));
        if ~exist(bin_dir, 'dir')
            mkdir(bin_dir);
        end

        InputFSO_frames = cell(N_frames_per_bin, 1);
        data_tx_3frame = [];
        frame_info = struct([]);

        for frame_idx = 1:N_frames_per_bin

            global_frame_id = (bin_idx - 1) * N_frames_per_bin + frame_idx;

            % 每帧、每调制使用独立种子；子载波循环内继续递增
            TX.BIT.seed = 100 + m * 1000000 + global_frame_id * 1000;

            %% ---------- Generate one OFDM frame ----------

            S.txofdm = zeros(ofdm.NumberOfCarriers, SIG.nSyms);

            for sc = 1:ofdm.NumberOfCarriers

                TX.BIT.txBits1 = Tx_generateBits( ...
                    SIG.nSyms, ...
                    TX.QAM.M, ...
                    TX.QAM.nPol, ...
                    TX.BIT);

                % 均匀 QAM 映射
                [S.tx, txSyms] = Tx_QAM(TX.QAM, TX.BIT.txBits1); %#ok<ASGLU>

                TX.BIT.seed = TX.BIT.seed + 1;

                % 与原 OFDM 链路一致的载波缩放
                S.txofdm(sc, :) = carrier_scale * S.tx;
            end

            % data_tx: [nSyms × nCarriers]
            S.txofdm = S.txofdm.';
            data_tx = S.txofdm;

            %% ---------- OFDM modulation ----------

            S.txofdm_time = OFDM(S.txofdm, ofdm, SIG.nSyms);

            % 原链路注释：16G -> 80G
            S.txSC = resample(S.txofdm_time, 80e9, 16e9);

            InputFSO = S.txSC(:);

            %% ---------- 256-point alignment per frame ----------

            alen = ceil(length(InputFSO) / 256) * 256;
            if length(InputFSO) < alen
                InputFSO = [InputFSO; zeros(alen - length(InputFSO), 1)];
            end

            InputFSO_frames{frame_idx} = InputFSO;

            if isempty(data_tx_3frame)
                data_tx_3frame = zeros(size(data_tx,1), size(data_tx,2), N_frames_per_bin);
            end
            data_tx_3frame(:, :, frame_idx) = data_tx;

            frame_info(frame_idx).global_frame_id = global_frame_id;
            frame_info(frame_idx).frame_idx_in_bin = frame_idx;
            frame_info(frame_idx).frame_len_samples = length(InputFSO);
            frame_info(frame_idx).mat_file = sprintf('frame_%04d.mat', frame_idx);

            frame_mat_file = fullfile(bin_dir, sprintf('frame_%04d.mat', frame_idx));
            save(frame_mat_file, 'data_tx', 'InputFSO', 'mname', 'bits', 'Mq', ...
                'bin_idx', 'frame_idx', 'global_frame_id', 'carrier_scale', ...
                'ofdm', 'SIG');
        end

        %% ---------- Concatenate 3 frames into one bin ----------

        InputFSO_3frame = [];
        gap = zeros(gap_len_samples, 1);

        for frame_idx = 1:N_frames_per_bin
            if frame_idx > 1 && gap_len_samples > 0
                InputFSO_3frame = [InputFSO_3frame; gap]; %#ok<AGROW>
            end
            InputFSO_3frame = [InputFSO_3frame; InputFSO_frames{frame_idx}]; %#ok<AGROW>
        end

        combined_mat_file = fullfile(bin_dir, sprintf('sig_%04d.mat', bin_idx));
        combined_txt_file = fullfile(bin_dir, sprintf('sig_%04d.txt', bin_idx));
        combined_bin_file = fullfile(bin_dir, sprintf('sig_%04d.bin', bin_idx));

        save(combined_mat_file, 'InputFSO_3frame', 'InputFSO_frames', ...
            'data_tx_3frame', 'frame_info', 'mname', 'bits', 'Mq', ...
            'bin_idx', 'N_frames_per_bin', 'gap_len_samples', ...
            'carrier_scale', 'ofdm', 'SIG');

        write_complex_txt(combined_txt_file, InputFSO_3frame);
        write_complex_bin_float32_iq(combined_bin_file, InputFSO_3frame);

        fprintf('  %s bin_%04d saved: 3 frames, %d samples total\n', ...
            mname, bin_idx, length(InputFSO_3frame));
    end
end

fprintf('\n============================================\n');
fprintf('All done. Uniform QAM TX generated.\n');
fprintf('Output root:\n  %s\n', out_root);
fprintf('Per modulation: %d bins × %d frames = %d frames\n', ...
    N_bins_per_mod, N_frames_per_bin, N_bins_per_mod * N_frames_per_bin);
fprintf('============================================\n');

%% ===================== Local functions =====================

function write_complex_txt(filename, x)
    % 两列 ASCII: real imag
    fid = fopen(filename, 'w');
    if fid < 0
        error('Cannot open file for writing: %s', filename);
    end
    x = x(:);
    fprintf(fid, '%.12e %.12e\n', [real(x).'; imag(x).']);
    fclose(fid);
end

function write_complex_bin_float32_iq(filename, x)
    % interleaved float32: real1 imag1 real2 imag2 ...
    fid = fopen(filename, 'wb');
    if fid < 0
        error('Cannot open file for writing: %s', filename);
    end
    x = x(:);
    fwrite(fid, [real(x).'; imag(x).'], 'float32');
    fclose(fid);
end
