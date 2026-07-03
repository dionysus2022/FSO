%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%盲相位搜索算法
%bps(数据输入, 测试相位个数, 考虑前后N个点, 理想星座点, 旋转范围)
function [data_bps_out, bps_out] = bps_(data_bps_in, B, N,sigConst, phaseLimit,Nosym,noFFT,NullsubcInd,loop)
data_bps_in=data_bps_in.';
data_out=zeros(noFFT,Nosym);
BPS_PN= zeros(noFFT,1);
for nn=1:noFFT
    % 调整行列，必须输入M*1
    data_phi = data_bps_in(nn,:).';
   % data_phi(NullsubcInd)=0;
    b= -B/2:B/2- 1;
    phi_b = b/B* phaseLimit; %测试相位，BPSK可以在正负phaseLimit/2的范围转
    dataLength = length(data_phi);
    data_rt = zeros(dataLength, B); %旋转后的信号

    %计算旋转后的星座点与MQAM中M个理想星座点的距离dkb
    % 将信号分别乘上不同相位
    for m = 1 : B
        data_rt(:, m) = data_phi.*exp(-1i * phi_b(m));
    end

    complex_mapping = sigConst;

    yk_b = zeros(dataLength, B);
    dk_b = zeros(dataLength, B);
    % 获取最理想星座点，以及最小平方距离
    for n=1 :  dataLength
        for m = 1: B
            [min_dkb, idx] = min(abs(data_rt(n, m)-complex_mapping).^2);  %与每个理想星座点的距离
            dk_b(n, m) = min_dkb; %最小距离
            yk_b(n, m) = complex_mapping(idx); %最小距离对应的理想星座点
        end
    end
    % 前后2N个平方距离求和，避免突发噪声
    % (N的最佳取值取决于激光器线宽和符号速率的商，一般取10),实际上是将散点各个最小距离加起来然后去比大小，目的是找到总体最小距离的那个旋转角度
    sk_b = dk_b;
    for m = 1 :  B
        for n = 1 : dataLength
            if (n < N) 
                sk_b(n, m) = sum(dk_b(1 : n+N, m));
            elseif (n >= N && n < dataLength - N)
                sk_b(n, m) = sum(dk_b(n-N+1 : n+N,  m));
            else
                sk_b(n, m) = sum(dk_b(n-N+1 : end,  m));
            end
        end
    end

    %每个测试相位都求出了一个值，值最小的那个测试相位ph_b,我们认为这2N个符号的相偏就是ph_b,然后再处理接下来的2N个数据。
    
    % 获取估计相位
    for m = 1:B
        skb(m)=sum(sk_b(:,m));
    end
    [~, ind] = min(skb);
    BPS_PN(nn)=phi_b(ind);%其实最后获得的是总体(是每个子载波上取一个星座点)旋转角度，并不是每个点都各自可以旋转不同角度


    % for n = 1 : dataLength
    %     skb = sk_b(n, :);
    %     [~, ind] = min(skb);
    %     BPS_PN(nn,n) = phi_b(ind);


        % if nn>1  %解决跳变问题
        %     if BPS_PN(nn)-BPS_PN(nn-1) > phaseLimit/2  %此门限可以看情况调整
        %         BPS_PN(nn:end) = BPS_PN(nn:end) - phaseLimit;  %可以看情况调整
        %     elseif BPS_PN(nn)-BPS_PN(nn-1) < -phaseLimit/2
        %         BPS_PN(nn:end) = BPS_PN(nn:end) + phaseLimit;
        %     end
        % end
        % 
    % BPS_PN = BPS_PN.';  

    
   
    % figure,plot(data_out(nn,:),'b.'), title('BPS sym out'),axis([-4 4,-4 4]);

end

for nn=2:1: noFFT

    %解决跳变问题
    if BPS_PN(nn)-BPS_PN(nn-1) > phaseLimit/2  %此门限可以看情况调整
        BPS_PN(nn:end) = BPS_PN(nn:end) - phaseLimit;  %可以看情况调整
    elseif BPS_PN(nn)-BPS_PN(nn-1) < -phaseLimit/2
        BPS_PN(nn:end) = BPS_PN(nn:end) + phaseLimit;
    end
end
for nn=1:1:noFFT
 data_phi = data_bps_in(nn,:).';
    data_out(nn,:)= data_phi.*exp(-1i * (BPS_PN(nn))).';
end
    data_bps_out = data_out; %经过相位恢复的信号
    bps_out = BPS_PN;  %每个点补偿的相位，要尽量平滑，不要跳变

    figure,plot(BPS_PN,'r'), title(['BPS phase estimate,第',num2str(loop),'帧']);
    figure,plot(data_bps_out,'b.'), title(['BPS out,第',num2str(loop),'帧']),axis(1.5*[-1 1,-1 1]);

end
