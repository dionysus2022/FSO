% 基于延时相关算法的edge搜索
% 理想时输出帧头位置，帧头连0处的第一个0位置
function [detected_packet, edge_index] = packet_edge_power_dect(rx_signal,zeros_head)
    Bit_width = 15;
    tx_signal_quantized = rx_signal * (2^(Bit_width-1)-1)/max(abs(rx_signal));
    tx_signal_round = round(tx_signal_quantized); % round函数是一个四舍五入的取整函数
    search_range=length(rx_signal)/2;

    for n=1:search_range
        P(n)=sum(abs(tx_signal_round(n:n+zeros_head-1)));  %1:80,2:81,3:82,...,8160:8239
    end

    figure(8)
    plot(1:search_range,P(1:search_range),'-b');
    title('找帧头连零处','fontsize',16,'fontname','宋体')

    [min_p,edge_index] = min(P);
    disp(strcat('Pack Edge Index is:  ', num2str(edge_index)));

    % 把帧头前的数据（Noise）截掉
    detected_packet = rx_signal(edge_index: length(rx_signal));

    figure(9)
    plot(real(detected_packet),'-b');
    title('除去帧头前的数据','fontsize',16,'fontname','宋体')
end