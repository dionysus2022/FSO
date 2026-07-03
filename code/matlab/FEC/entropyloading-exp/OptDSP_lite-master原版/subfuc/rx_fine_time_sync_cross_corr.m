% Fine time synchronization，输出第一个训练序列的开始位置
function [fine_time_est,data_df,max_peak_long] = rx_fine_time_sync_cross_corr(syn1_signal, LtrsCPLength,LongTrainSym,zeros_head,plot_en)

    % Timing search window size
    % start_search = 64*2; 128
    % end_search = start_search + 5 * 16;  208
	start_search = 1;
	end_search = zeros_head+LtrsCPLength+length(LongTrainSym)-16; %搜索结束的点不能太长，因为如果有两个训练序列，当索引到第一个训练序列结束，第二个开始时，刚好和第二个训练序列相关程度最大，输出的是75+16+256=348，而不是第75+16=92 

    % Next generate the two long training symbols
	ifft_LongTrainSym = ifft(LongTrainSym);
	time_corr_long = zeros(1, end_search - start_search + 1); % 1×136的0矩阵

    %Calculate cross correlation 计算互相关
	for idx = 1 : end_search - start_search + 1
        time_corr_long(idx) = ...
        sum((syn1_signal(idx+start_search-1 : idx + start_search-1 + length(LongTrainSym) - 1).* conj(ifft_LongTrainSym))); % syn1_signal(1:64).*conj(ifft_LongTrainSym)
    end

	%Find the biggest value
	[max_peak_long, long_search_idx] = max(abs(time_corr_long));
	data_df=sign(time_corr_long(long_search_idx)).*((time_corr_long(long_search_idx-1))-(time_corr_long(long_search_idx+1)));   
    
    figure(11)
	if plot_en==1
        plot(start_search:end_search,abs(time_corr_long),'-b')
        title('找训练序列的位置','fontsize',16,'fontname','宋体')
	end
	    fine_time_est = long_search_idx + start_search - 1;   
    
        
    %  正确的应是(64 + 16) * 2 + 16 * 2 + 1=193，而每次算出来都是195，不知为何
    %  fine_time_est=fine_time_est-2; % ??? 人为减掉2,成就193~~
  
end