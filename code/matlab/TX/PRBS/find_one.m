function pos = find_ones(vec)
% 查找向量中值为1的元素位置（忽略100等占位符）
pos = [];
for i = 1:length(vec)
    if vec(i) == 1  % 只识别值为1的元素
        pos = [pos, i];
    end
end
end