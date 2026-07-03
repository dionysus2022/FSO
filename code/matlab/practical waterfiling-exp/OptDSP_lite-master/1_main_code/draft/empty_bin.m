%% 需求：共200个bin文件，文件名1.bin~200.bin
%% 8个子文件夹 sub01~sub08，每个放25个
%% sub01：1~25.bin；sub02：26~50.bin；……sub08：176~200.bin
clear; clc;

totalFile = 225;
filePerSub = 25;
subNum = totalFile / filePerSub;

for subIdx = 1 : subNum
    % 子文件夹名 sub01,sub02...sub08
    subDir = sprintf('sub%02d', subIdx);
    if ~exist(subDir, 'dir')
        mkdir(subDir);
        fprintf('创建文件夹：%s\n', subDir);
    end
    
    % 计算当前文件夹存放文件编号区间
    startId = (subIdx - 1) * filePerSub + 1;
    endId   = subIdx * filePerSub;
    
    % 循环生成该区间所有空bin
    for fidNum = startId : endId
        filePath = fullfile(subDir, sprintf('%d.bin', fidNum));
        fid = fopen(filePath, 'wb');
        if fid ~= -1
            fclose(fid);
        else
            warning('创建失败：%s', filePath);
        end
    end
    fprintf('  %s 已生成 %d～%d.bin\n', subDir, startId, endId);
end

fprintf('\n全部完成，总计 %d 个bin文件\n', totalFile);