% =========================================================================
% 信号调制格式识别 - 模型训练与测试
% 数据来源: dataset_cdm_exp_all
% =========================================================================

clear all; close all; clc;

%% ===== 配置参数 =====
data_dir = 'D:\matlab\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\dataset_cdm_exp_all';

% 调制格式映射
mod_map = containers.Map({1, 2, 4, 6, 8}, {'BPSK', 'QPSK', '16QAM', '64QAM', '256QAM'});

% 调制格式对应的颜色（用于绘图）
colors = {[0 0.4470 0.7410], [0.8500 0.3250 0.0980], [0.9290 0.6940 0.1250], ...
          [0.4940 0.1840 0.5560], [0.4660 0.6740 0.1880]};

fprintf('==================================================\n');
fprintf('信号调制格式识别 - 模型训练与测试\n');
fprintf('数据目录: %s\n', data_dir);
fprintf('==================================================\n\n');

%% ===== 1. 加载数据 =====
fprintf('正在加载数据...\n');

% 获取所有 .mat 文件
all_files = dir(fullfile(data_dir, '*.mat'));
num_files = length(all_files);

if num_files == 0
    error('未找到任何 .mat 文件！');
end

fprintf('找到 %d 个数据文件\n', num_files);

% 初始化存储
features = [];
labels = [];
filenames = {};

% 逐个加载文件
for i = 1:num_files
    filepath = fullfile(data_dir, all_files(i).name);
    
    try
        % 加载数据
        data = load(filepath);
        
        % 获取变量名（排除通常的元数据变量）
        var_names = fieldnames(data);
        
        % 查找特征数据（通常是数值矩阵）
        feature_data = [];
        label_value = [];
        
        for j = 1:length(var_names)
            var_name = var_names{j};
            
            % 跳过元数据变量
            if ismember(var_name, {'__header__', '__version__', '__globals__'})
                continue;
            end
            
            var = data.(var_name);
            
            % 如果是数值数组且维度合适，则认为是特征
            if isnumeric(var) && ~isempty(var)
                if ismatrix(var) && size(var, 2) > 1
                    % 可能是特征矩阵
                    feature_data = var;
                elseif size(var, 2) == 1
                    % 可能是标签
                    label_value = var;
                end
            elseif isstruct(var)
                % 检查结构体字段
                fields = fieldnames(var);
                for k = 1:length(fields)
                    field_data = var.(fields{k});
                    if isnumeric(field_data) && ismatrix(field_data) && size(field_data, 2) > 1
                        feature_data = field_data;
                        break;
                    end
                end
            end
        end
        
        % 如果没找到特征数据，尝试直接使用第一个数值变量
        if isempty(feature_data) && length(var_names) > 0
            for j = 1:length(var_names)
                if ~ismember(var_names{j}, {'__header__', '__version__', '__globals__'})
                    var = data.(var_names{j});
                    if isnumeric(var) && ismatrix(var) && ~isempty(var)
                        feature_data = var;
                        break;
                    end
                end
            end
        end
        
        % 从文件名提取标签 (exp_frame_XXX_mod_Y.mat)
        [~, name, ~] = fileparts(all_files(i).name);
        parts = strsplit(name, '_');
        if length(parts) >= 4
            mod_str = parts{4};
            if strncmp(mod_str, 'mod_', 4)
                label_value = str2double(mod_str(5:end));
            end
        end
        
        % 存储数据
        if ~isempty(feature_data)
            features = [features; feature_data];
            labels = [labels; repmat(label_value, size(feature_data, 1), 1)];
            filenames{end+1} = all_files(i).name;
        end
        
        if mod(i, 100) == 0 || i == num_files
            fprintf('已加载: %d/%d\r', i, num_files);
        end
        
    catch ME
        fprintf('警告: 加载文件 %s 失败: %s\n', all_files(i).name, ME.message);
    end
end

fprintf('\n');

% 确保 features 是 2D 矩阵
if size(features, 1) < size(features, 2)
    features = features.';
end

fprintf('加载完成！\n');
fprintf('特征矩阵大小: %d x %d\n', size(features, 1), size(features, 2));
fprintf('标签数量: %d\n\n', length(labels));

%% ===== 2. 统计各类别样本数 =====
fprintf('==================================================\n');
fprintf('数据集统计:\n');
fprintf('==================================================\n');

mod_classes = [1, 2, 4, 6, 8];
for i = 1:length(mod_classes)
    mod_val = mod_classes(i);
    mod_name = mod_map(mod_val);
    count = sum(labels == mod_val);
    fprintf('%s (mod_%d): %d 个样本\n', mod_name, mod_val, count);
end
fprintf('总计: %d 个样本\n\n', length(labels));

%% ===== 3. 数据预处理 =====
fprintf('正在进行数据预处理...\n');

% 处理 NaN 和 Inf
features(isnan(features)) = 0;
features(isinf(features)) = 0;

% 归一化
feature_mean = mean(features, 1);
feature_std = std(features, 0, 1);
feature_std(feature_std == 0) = 1;  % 避免除零
features_normalized = (features - feature_mean) ./ feature_std;

fprintf('预处理完成！\n\n');

%% ===== 4. 划分训练集和测试集 =====
fprintf('==================================================\n');
fprintf('划分训练集和测试集\n');
fprintf('==================================================\n');

train_ratio = 0.7;  % 70% 训练，30% 测试
num_samples = length(labels);

% 随机打乱索引
rng(42);  % 设置随机种子以保证可重复性
indices = randperm(num_samples);

% 计算分割点
split_idx = floor(num_samples * train_ratio);
train_indices = indices(1:split_idx);
test_indices = indices(split_idx+1:end);

X_train = features_normalized(train_indices, :);
y_train = labels(train_indices);
X_test = features_normalized(test_indices, :);
y_test = labels(test_indices);

fprintf('训练集: %d 个样本\n', length(train_indices));
fprintf('测试集: %d 个样本\n\n', length(test_indices));

%% ===== 5. 训练分类模型 =====
fprintf('==================================================\n');
fprintf('训练分类模型\n');
fprintf('==================================================\n\n');

% 使用多种分类器进行比较
results = struct();

%% 5.1 KNN 分类器
fprintf('--- KNN 分类器 ---\n');
knn_model = fitcknn(X_train, y_train, 'NumNeighbors', 5, 'Distance', 'euclidean');
knn_pred = predict(knn_model, X_test);
knn_acc = sum(knn_pred == y_test) / length(y_test);
fprintf('KNN 准确率: %.2f%%\n\n', knn_acc * 100);
results.KNN = struct('model', knn_model, 'pred', knn_pred, 'accuracy', knn_acc);

%% 5.2 SVM 分类器
fprintf('--- SVM 分类器 ---\n');
svm_model = fitcecoc(X_train, y_train, 'Learners', templateSVM('KernelFunction', 'rbf'));
svm_pred = predict(svm_model, X_test);
svm_acc = sum(svm_pred == y_test) / length(y_test);
fprintf('SVM 准确率: %.2f%%\n\n', svm_acc * 100);
results.SVM = struct('model', svm_model, 'pred', svm_pred, 'accuracy', svm_acc);

%% 5.3 随机森林
fprintf('--- 随机森林 ---\n');
rf_model = fitcensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', 100);
rf_pred = predict(rf_model, X_test);
rf_acc = sum(rf_pred == y_test) / length(y_test);
fprintf('随机森林准确率: %.2f%%\n\n', rf_acc * 100);
results.RF = struct('model', rf_model, 'pred', rf_pred, 'accuracy', rf_acc);

%% ===== 6. 评估结果 =====
fprintf('==================================================\n');
fprintf('模型性能对比\n');
fprintf('==================================================\n');
fprintf('%-15s | 准确率\n', '模型');
fprintf('---------------------------\n');
fprintf('%-15s | %.2f%%\n', 'KNN', knn_acc * 100);
fprintf('%-15s | %.2f%%\n', 'SVM', svm_acc * 100);
fprintf('%-15s | %.2f%%\n', '随机森林', rf_acc * 100);
fprintf('---------------------------\n\n');

% 选择最佳模型
[best_acc, best_model_name] = max([knn_acc, svm_acc, rf_acc]);
best_models = {'KNN', 'SVM', '随机森林'};
fprintf('最佳模型: %s (准确率: %.2f%%)\n\n', best_models{best_model_name}, best_acc * 100);

%% ===== 7. 混淆矩阵可视化 =====
fprintf('==================================================\n');
fprintf('生成混淆矩阵...\n');
fprintf('==================================================\n\n');

% 使用最佳模型的预测结果
switch best_model_name
    case 1
        best_pred = knn_pred;
    case 2
        best_pred = svm_pred;
    case 3
        best_pred = rf_pred;
end

% 创建混淆矩阵
figure('Position', [100, 100, 800, 600]);
mod_names = {'BPSK', 'QPSK', '16QAM', '64QAM', '256QAM'};
C = confusionchart(y_test, best_pred, 'RowSummary', 'total-normalized');
C.Title = sprintf('混淆矩阵 - %s (准确率: %.2f%%)', best_models{best_model_name}, best_acc * 100);
C.XLabel = '预测标签';
C.YLabel = '真实标签';

% 设置标签
C.ClassLabels = mod_names;

%% ===== 8. 各类别性能分析 =====
fprintf('\n==================================================\n');
fprintf('各类别分类性能 (使用 %s)\n', best_models{best_model_name});
fprintf('==================================================\n');

for i = 1:length(mod_classes)
    mod_val = mod_classes(i);
    mod_name = mod_map(mod_val);
    
    % 计算该类别的精确率、召回率、F1
    true_pos = sum(best_pred == mod_val & y_test == mod_val);
    false_pos = sum(best_pred == mod_val & y_test ~= mod_val);
    false_neg = sum(best_pred ~= mod_val & y_test == mod_val);
    
    precision = true_pos / (true_pos + false_pos);
    recall = true_pos / (true_pos + false_neg);
    f1 = 2 * precision * recall / (precision + recall);
    
    fprintf('%s: 精确率=%.2f%%, 召回率=%.2f%%, F1=%.2f%%\n', ...
            mod_name, precision*100, recall*100, f1*100);
end

%% ===== 9. 保存结果 =====
fprintf('\n==================================================\n');
fprintf('保存结果...\n');
fprintf('==================================================\n');

% 保存模型
save_dir = fullfile(data_dir, '..', 'model_results');
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% 保存最佳模型
switch best_model_name
    case 1
        save(fullfile(save_dir, 'best_model_KNN.mat'), 'knn_model');
    case 2
        save(fullfile(save_dir, 'best_model_SVM.mat'), 'svm_model');
    case 3
        save(fullfile(save_dir, 'best_model_RF.mat'), 'rf_model');
end

% 保存处理后的数据
save(fullfile(save_dir, 'train_test_data.mat'), ...
     'X_train', 'y_train', 'X_test', 'y_test', ...
     'feature_mean', 'feature_std');

% 保存分类报告
report = struct();
report.best_model = best_models{best_model_name};
report.best_accuracy = best_acc;
report.KNN_accuracy = knn_acc;
report.SVM_accuracy = svm_acc;
report.RF_accuracy = rf_acc;
report.num_train = length(train_indices);
report.num_test = length(test_indices);
save(fullfile(save_dir, 'classification_report.mat'), 'report');

fprintf('结果已保存到: %s\n', save_dir);
fprintf('\n==================================================\n');
fprintf('训练与测试完成！\n');
fprintf('==================================================\n');
