function [power_alloc,bit_alloc,bit_alloc_theory] = fading_waterFilling(NumberOfCarriers,ch_fading)
%% 初始化参数

N_subc = NumberOfCarriers;                              %子载波数目
BER=1e-3;                                %目标误比特率
psgap=10.^(1.53./10)
%gap=-2/3*log(5*BER);                %psgap是1.53dB,减去1.53db,化成倍数就是除以psgap
%gap = ((qfuncinv(BER/4))^2)/3
gap =1;
P_av=123/N_subc;                        %每个子载波的平均功率
Ptsum=P_av*N_subc;                 %总发射功率
% SNR_sub=20*ones(1,N_subc)%30+30*(rand(1,N_subc)-0.5);   %平均信噪比20dB
% noise=P_av./10.^(SNR_sub./10);             %每个子信道的噪声功率
%% 仿真瑞利衰落
 %   hn=ch_fading;         %子载波增益
 %  %hn=ones(1,127);
 % %hn=random('rayleigh',1,1,N_subc);
gn=10.^(ch_fading./10);
%% 初始化分配的功率
[gain_sorted,dt]=sort(gap./gn);                 %对增益进行排序
for p=length(gain_sorted):-1:1 
    T_P=(Ptsum+sum(gain_sorted(1:p)))/p;   %计算注水线
    Input_Power=T_P-gain_sorted;                 %进行功率分配
    Pt=Input_Power(1:p); 
    if(Pt(:)>=0)                                     %所有子信道的功率都大于0，分配循环才break，出现了小于零的，一定是64个的末尾（因为是排好序的），于是下一个循环只剩下63，最后一个删掉了，循环之后这个子载波功率power_alloc令为0
        break                                          %删掉的不只是这个子载波的gn，对应索引dt(p)也删掉了
    end 
end 
power_alloc=zeros(1,N_subc); 
power_alloc(dt(1:p))=Pt;                        %注水法分配的功率
bit_alloc=log2(1+(power_alloc.*gn)./gap);    %注水法分配的比特
sum_bitideal=sum(bit_alloc)
bit_alloc_theory=log2(1+(power_alloc.*gn)./1.42);    %注水法分配的比特
sum_bitideal_theory=sum(bit_alloc_theory);
%实用注水
 bit_alloc=floor(bit_alloc);
power_alloc=(2.^bit_alloc-1).*(gap./gn);
p_tot=sum(power_alloc);
delta_pn=(2.^bit_alloc)*gap./gn;
[n_mini,i]=min(delta_pn);
while  p_tot+delta_pn(i)<=Ptsum
        bit_alloc(i)=bit_alloc(i)+1;
        p_tot=p_tot+delta_pn(i);
        power_alloc(i)=power_alloc(i)+ delta_pn(i);
        delta_pn(i)=2*delta_pn(i);
        [n_mini,i]=min(delta_pn);

end
p_tot=sum(power_alloc);
%power_alloc(1,:)=p_tot/N_subc;%功率调成平均分配
SNR=10*log10(power_alloc.*gn);
%和平均分配比较
energy1=sum(power_alloc);
bit_sum1=sum(bit_alloc);%这里可以发现，无论是实用注水还是理想注水，分配的比特量大差不差，因为gap确定，容量限就确定，所有能量都分配在这条线上，自然都符合这个log2(1+snr)的线
P_av=p_tot/N_subc


bit_alloc_aver=log2(1+(P_av.*gn)./gap);
prac_bit_alloc_aver=floor(bit_alloc_aver);
bit_sum2=sum(prac_bit_alloc_aver);
energy2=sum((2.^prac_bit_alloc_aver-1).*(gap./gn));
energy_eff1=energy1/bit_sum1;
energy_eff2=energy2/bit_sum2;
%% 绘图
% figure(1);
% subplot(311)
% stem(SNR_sub);
% ylabel('SNR average(dB)')
% set(gca,'ytick',[-10:10:30])
% subplot(312);
% stem(power_alloc);
% ylabel('Power allocation(W)')
% ylim([0 0.03])
% subplot(313);
% stem(bit_alloc);
% ylabel('Bit allocation(bit)')
% ylim([0 12])
% figure(1);
% stem(SNR_sub);
% ylabel('SNR average(dB)')
% set(gca,'ytick',[-10:10:30])
% figure();
% stem(power_alloc);
% ylabel('Power allocation(W)')
% 
% figure();
% stem(bit_alloc);
% ylabel('Bit allocation(bit)')
% ylim([0 12])

end

