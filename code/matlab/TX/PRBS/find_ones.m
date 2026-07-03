function [ pos ] = find_ones( x )

%作用：寻找向量中所有“1”的位置函数  这里写的代码将简单问题复杂化，并且存在错误
%% This function returns the position of the ones inide the input vector

% Input:

% - x = input vector

% Output:

% pos = vector containing the positions of the value 1 in x

i = 1;
while(x(i) ~= 1)
    i = i + 1;
end

x1 = x(i:end);
j = 1;
for i = length(x1):-1:1
    x2(j) = x1(i);
    j = j + 1;
end

pos = zeros(1,length(x2(x2 == 1)) - 1);
k = 1;
for i = 2:length(x2)
    if(x2(i) == 1)
        pos(k) = i - 1;
        k = k + 1;
    end
end



% function pos = find_ones(x)
%     pos = find(x == 1);  % 一行搞定！
% end

% 在你的FSO系统中，这个函数可能用于：
% 查找特定训练序列的位置
% 定位帧同步头
% 寻找LDPC编码中的校验位位置

% function pos = find_ones(x)
%     % 安全检查：确保x是向量
%     if isempty(x)
%         pos = [];
%         return;
%     end
% 
%     % 直接使用MATLAB内置的find函数
%     pos = find(x == 1);
% 
%     % 或者明确要求：忽略开头的0？
%     % 找到第一个1之后的所有1的位置
%     % first_one = find(x == 1, 1, 'first');
%     % pos = find(x(first_one:end) == 1) + first_one - 1;
% end