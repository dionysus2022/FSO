%% debug_256qam.m - 诊断256QAM数据文件问题
clear; close all; clc;
addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));
global PROG; PROG.showMessagesLevel = 2; initProg(); co = 1;

%% 1. 加载现有数据文件
fprintf('=== 1. 检查现有数据文件 ===\n');
mat_path = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\tx_1frame_5mod\256QAM\sub01\sig_0001.mat';
load(mat_path, 'data_tx');
sym_old = data_tx(:);

fprintf('data_tx size: %d x %d\n', size(data_tx,1), size(data_tx,2));
fprintf('Total symbols: %d\n', numel(sym_old));
fprintf('I range: [%.4f, %.4f]\n', min(real(sym_old)), max(real(sym_old)));
fprintf('Q range: [%.4f, %.4f]\n', min(imag(sym_old)), max(imag(sym_old)));
fprintf('Unique I values: %d\n', length(unique(round(real(sym_old),4))));
fprintf('Unique Q values: %d\n', length(unique(round(imag(sym_old),4))));
fprintf('Unique (I,Q) pairs: %d / %d symbols\n', ...
    length(unique([real(sym_old), imag(sym_old)], 'rows')), length(sym_old));

% 绘制现有的星座图
figure('Name', '现有256QAM数据');
plot(real(sym_old), imag(sym_old), 'b.', 'MarkerSize', 2);
axis equal; grid on;
title(sprintf('现有256QAM (sub01/sig_0001)\n%d个唯一星座点 / %d个符号', ...
    length(unique([real(sym_old), imag(sym_old)], 'rows')), length(sym_old)));
xlabel('I'); ylabel('Q');

%% 2. 重新生成1帧256QAM信号做对比
fprintf('\n=== 2. 用当前代码重新生成256QAM（1个OFDM符号，1个子载波）===\n');

% 设置256QAM参数
SIG.M = 256; SIG.symRate = 8e9/co; SIG.bitRate_net = 8e9; SIG.modulation = 'QAM';
SIG.rollOff = 0.25; SIG.nPol = 1; SIG.nSyms = 128; FEC_rate = 1; pilotRate = 1;
bits = 8; nBpS_net = bits;

TX.SIG = setSignalParams('symRate',SIG.symRate,'M',SIG.M,...
    'nPol',SIG.nPol,'nBpS',nBpS_net,'nSyms',SIG.nSyms,...
    'roll-off',SIG.rollOff,'modulation',SIG.modulation);
TX.QAM = QAM_config(TX.SIG); 
TX.BIT.source = 'randi'; TX.BIT.seed = 100;
TX.FEC.active = false; TX.FEC.rate = FEC_rate;
TX.PCS.method = 'CCDM';

fprintf('SIG.nBpS = %d\n', TX.SIG.nBpS);
fprintf('QAM.M = %d, QAM.nBpS = %d\n', TX.QAM.M, TX.QAM.nBpS);
fprintf('C (IQmap) size: %d, range: [%.2f, %.2f]\n', ...
    length(TX.QAM.IQmap), min(real(TX.QAM.IQmap)), max(real(TX.QAM.IQmap)));

% 生成1个子载波的信号
txBits = Tx_generateBits(SIG.nSyms, TX.QAM.M, TX.QAM.nPol, TX.BIT);
fprintf('txBits size: %d bits\n', numel(txBits));

[Stx, txSyms, TX.QAM_new] = Tx_ProbShaping(txBits, TX.QAM, TX.SIG, TX.FEC.rate);
fprintf('Stx size: %d symbols\n', numel(Stx));
fprintf('Stx I range: [%.4f, %.4f]\n', min(real(Stx)), max(real(Stx)));
fprintf('Stx Q range: [%.4f, %.4f]\n', min(imag(Stx)), max(imag(Stx)));
fprintf('Unique (I,Q) pairs in Stx: %d / %d\n', ...
    length(unique([real(Stx.'), imag(Stx.')], 'rows')), numel(Stx));

% 绘制新生成的星座图
figure('Name', '新生成256QAM (1个子载波)');
plot(real(Stx), imag(Stx), 'r.', 'MarkerSize', 4);
axis equal; grid on;
title(sprintf('新生成256QAM (1个子载波, nBpS=%d)\n%d个唯一星座点 / %d个符号', ...
    TX.SIG.nBpS, length(unique([real(Stx.'), imag(Stx.')], 'rows')), numel(Stx)));
xlabel('I'); ylabel('Q');

%% 3. 结论
fprintf('\n=== 诊断结论 ===\n');
n_unique_new = length(unique([real(Stx.'), imag(Stx.')], 'rows'));
n_unique_old = length(unique([real(sym_old), imag(sym_old)], 'rows'));

fprintf('现有数据文件中256QAM的唯一星座点数: %d (16x16=256才正常)\n', n_unique_old);
fprintf('新代码生成的256QAM唯一星座点数: %d\n', n_unique_new);

if n_unique_old < 50
    fprintf('>>> 结论: 现有256QAM数据是用旧代码生成的(nBpS_net=1)，需要重新运行tx_1frame_5mod.m\n');
else
    fprintf('>>> 结论: 现有256QAM数据看起来正常\n');
end

fprintf('\n请在figure窗口中对比两个星座图。\n');
fprintf('如果现有数据只有少量点(如4-8个)，说明需要重新生成。\n');
fprintf('如需重新生成，在MATLAB命令窗口运行:  tx_1frame_5mod\n');