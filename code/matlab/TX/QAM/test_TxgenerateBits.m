%% 测试 Tx_generateBits 函数（nSyms=100, M=16, nPol=2）
%这段测试代码依赖 3 个自定义函数：Tx_generateBits.m、QAM_PRBSgenerator.m、PRBS_generator.m。
% 需要把这 3 个函数保存为独立的.m文件，和测试脚本放在同一文件夹。

clear; clc;  % 清空工作区、清屏

% 1. 定义基础输入参数
nSyms = 100;   % 符号数
M = 16;        % 16QAM
nPol = 2;      % 双偏振

% 2. 测试三种比特源类型
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 类型1：randi（默认，随机数生成）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('===== 测试比特源类型：randi =====\n');
BIT_randi.source = 'randi';
BIT_randi.seed = 123;  % 固定种子，保证结果可复现
txBits_randi = Tx_generateBits(nSyms,M,nPol,BIT_randi);
% 输出关键信息
fprintf('randi模式 - 比特矩阵维度：%d行 × %d列\n', size(txBits_randi,1), size(txBits_randi,2));
fprintf('randi模式 - 前5列比特（偏振态1/2）：\n');
disp(txBits_randi(:,1:5));  % 显示前5列比特

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 类型2：PRBS（基础PRBS生成）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n===== 测试比特源类型：PRBS =====\n');
BIT_prbs.source = 'PRBS';
BIT_prbs.seed = 123;  % 固定种子
txBits_prbs = Tx_generateBits(nSyms,M,nPol,BIT_prbs);
% 输出关键信息
fprintf('PRBS模式 - 比特矩阵维度：%d行 × %d列\n', size(txBits_prbs,1), size(txBits_prbs,2));
fprintf('PRBS模式 - 前5列比特（偏振态1/2）：\n');
disp(txBits_prbs(:,1:5));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 类型3：PRBS-QAM（调用QAM_PRBSgenerator）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n===== 测试比特源类型：PRBS-QAM =====\n');
BIT_prbsqam.source = 'PRBS-QAM';
BIT_prbsqam.seed = 123;    % 必须指定种子
BIT_prbsqam.degree = 9;    % PRBS阶数（2^9-1=511，足够400比特）
txBits_prbsqam = Tx_generateBits(nSyms,M,nPol,BIT_prbsqam);
% 输出关键信息
fprintf('PRBS-QAM模式 - 比特矩阵维度：%d行 × %d列\n', size(txBits_prbsqam,1), size(txBits_prbsqam,2));
fprintf('PRBS-QAM模式 - 前5列比特（偏振态1/2）：\n');
disp(txBits_prbsqam(:,1:5));

%% 附：需要把 Tx_generateBits.m 和 QAM_PRBSgenerator.m 放在同一文件夹！