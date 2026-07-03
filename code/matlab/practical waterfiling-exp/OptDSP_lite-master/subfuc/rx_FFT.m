%FFT operation, downsample and separate the data
function [freq_tr_syms, data_syms_1] = rx_FFT(sync_signal,numFFT,CPLength,numsym,Ltrs_num)

    long_training_end = numFFT*Ltrs_num;  % long_training_end = 64*1     长训练序列没有CP吗？？？
    ofdm_symbol_len = numFFT + CPLength;  % ofdm_symbol_len = 64+12 = 80
        
    % Long Training symbols
    long_tr_syms = sync_signal(1: long_training_end);        % 本帧数据中的长训练序列
    long_tr_syms_1 = reshape(long_tr_syms,numFFT,Ltrs_num)'; % matrix size:1×64
    long_tr_syms_2 = conj(long_tr_syms_1);   % 把longtrain分为两行(本来就是两个周期重复的长训练序列)
    % To frequency domain
    freq_tr_syms = fft(long_tr_syms_2,[],2); % 每行单独进行FFT，2为维度

    % Take data symbols 真正的数据段,把training和后面的noise都截掉了
    % Cut to multiple of symbol period
    data_syms_1 = sync_signal(long_training_end + 1: long_training_end + ofdm_symbol_len * numsym); % data_syms_1 = sync_signal(512+1:512+272*1024)
    % Remove guard intervals
    data_syms_2 = conj(reshape(data_syms_1,ofdm_symbol_len,numsym)'); % matrix size: 100×80
    
    
    data_syms_2(:, 1 : CPLength) = []; % 删掉CP
    % To frequency domain/Perform fft
    freq_data_syms = fft(data_syms_2, [], 2); % 对每一行进行FFT，freq_data_syms size应是100*64
end