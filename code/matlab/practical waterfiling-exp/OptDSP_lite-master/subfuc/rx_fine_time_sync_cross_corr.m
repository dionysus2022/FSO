% Fine time synchronization�������һ��ѵ�����еĿ�ʼλ��
function [fine_time_est,data_df,max_peak_long] = rx_fine_time_sync_cross_corr(syn1_signal, LtrsCPLength,LongTrainSym,zeros_head,plot_en)

    % Timing search window size
    % start_search = 64*2; 128
    % end_search = start_search + 5 * 16;  208
	start_search = 1;
	end_search = zeros_head+LtrsCPLength+length(LongTrainSym)-16; %���������ĵ㲻��̫������Ϊ���������ѵ�����У�����������һ��ѵ�����н������ڶ�����ʼʱ���պú͵ڶ���ѵ��������س̶�����������75+16+256=348�������ǵ�75+16=92 

    % Next generate the two long training symbols
	ifft_LongTrainSym = ifft(LongTrainSym);
	time_corr_long = zeros(1, end_search - start_search + 1); % 1��136��0����

    %Calculate cross correlation ���㻥���
	for idx = 1 : end_search - start_search + 1
        time_corr_long(idx) = ...
        sum((syn1_signal(idx+start_search-1 : idx + start_search-1 + length(LongTrainSym) - 1).* conj(ifft_LongTrainSym))); % syn1_signal(1:64).*conj(ifft_LongTrainSym)
    end

	%Find the biggest value
	[max_peak_long, long_search_idx] = max(abs(time_corr_long));
	data_df=sign(time_corr_long(long_search_idx)).*((time_corr_long(long_search_idx-1))-(time_corr_long(long_search_idx+1)));   
    
    % figure(11)
	if plot_en==1
        plot(start_search:end_search,abs(time_corr_long),'-b')
        title('��ѵ�����е�λ��','fontsize',16,'fontname','����')
	end
	    fine_time_est = long_search_idx + start_search - 1;   
    
        
    %  ��ȷ��Ӧ��(64 + 16) * 2 + 16 * 2 + 1=193����ÿ�����������195����֪Ϊ��
    %  fine_time_est=fine_time_est-2; % ??? ��Ϊ����2,�ɾ�193~~
  
end