%辅助函数，是QAM-config兜底工具，从预制的 MAT 数据文件
% 中加载自定义 QAM 星座图（比如 MATLAB 内置函数不支持的
% 十字形 32QAM、512QAM），保证 QAM_config 能处理所有指定的调制格式
function [const,symbolMap] = QAM_loadConstellation(MF_ID)
%输入：MF_ID（调制格式标识字符串，如32QAM_cross、8QAM_rect）；
%输出：const（加载的星座图复数矩阵）、symbolMap（符号映射表）
%需要提前准备好以 MF_ID 命名的 .mat 数据文件（如32QAM_cross.mat），
% 文件中必须包含 Constellation（星座点）和 SymbolMapping（符号映射）两个变量。

% Last Update: 13/02/2018


%% Load Constellation
C = load(MF_ID); %加载指定的.mat数据文件
%核心作用：读取以 MF_ID 为文件名的 MAT 数据文件（后缀.mat可省略），并将文件中的所有变量封装到结构体 C 中。

const = C.Constellation; %提取星座图数据
%核心作用：从加载的结构体 C 中，取出名为 Constellation 的变量，赋值给输出参数 const

symbolMap = C.SymbolMapping;
%核心作用：从结构体 C 中取出名为 SymbolMapping 的变量，赋值给输出参数 symbolMap。
%SymbolMapping 存储的是“符号索引-星座点位置”的映射关系