function [power_alloc,gmi_snr,lambda_snr,order] = entropyloading_NGMI(NumberOfCarriers,ch_fading,nSpS,SIG,TX,t)
%% 初始化参数

N_subc = NumberOfCarriers;                              %子载波数目
BER=0.164;                                %目标误比特率
%BER=1e-5
psgap=10.^(1.53./10)
%gap=-2/3*log(5*BER);                %psgap是1.53dB,减去1.53db,化成倍数就是除以psgap
%gap = ((qfuncinv(BER/4))^2)/3
gap =1;
P_av=123/N_subc;                        %每个子载波的平均功率
Ptsum=P_av*N_subc;                 %总发射功率
SNR_sub=18*ones(1,N_subc)%30+30*(rand(1,N_subc)-0.5);   %平均信噪比20dB
noise=P_av./10.^(SNR_sub./10);             %每个子信道的噪声功率
%% 仿真瑞利衰落
%    hn=ch_fading;         %子载波增益
% 
% gn=hn.^2./noise;
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
bit_alloc=log2(1+(gn)./gap);    %注水法分配的比特
%黄金分割查找
a=sum(bit_alloc);
%power_alloc=ones(1,N_subc); 
SNR=ch_fading;
save('power_alloc.mat','power_alloc');
save('shannon cap.mat','bit_alloc');
load('lammda LUT.mat');

NGMI_thres=0.92;

for snr_iter = 1:N_subc
  %  if   isinf(SNR(snr_iter)) 
  %       continue
  % elseif (17<SNR(snr_iter))
  %       order(snr_iter)=1024;
  %       stop=unnamed(8,floor((SNR(snr_iter)-(-15))/0.5)+2);%由于lambda查找表从-15到35db，所以算距离-15差多少，然后除以0.5这个步长，再加上1这个数组初始索引。
  %       begin=unnamed(8,floor((SNR(snr_iter)-(-15))/0.5)+1);
  % elseif (11<SNR(1,snr_iter))&&(SNR(snr_iter)<=17 )
  %       order(snr_iter)=256;
  %       stop=unnamed(6,floor((SNR(snr_iter)-(-15))/0.5)+2);
  %       begin=unnamed(6,floor((SNR(snr_iter)-(-15))/0.5)+1);
  % elseif (5<SNR(snr_iter))&&(SNR(snr_iter)<=11 )
  %       order(snr_iter)=64;
  %       stop=unnamed(4,floor((SNR(snr_iter)-(-15))/0.5)+2);
  %       begin=unnamed(4,floor((SNR(snr_iter)-(-15))/0.5)+1);
  % elseif  (-15<=SNR(snr_iter))&&(SNR(snr_iter)<=5 )
  %      order(snr_iter)=16;
  %       stop=unnamed(2,floor((SNR(snr_iter)-(-15))/0.5)+2);
  %       begin=unnamed(2,floor((SNR(snr_iter)-(-15))/0.5)+1);
  %  elseif SNR(1,snr_iter)<-15
  %      order(snr_iter)=2;
  %      stop=1.6;%如果信噪比小于-15，则不进行PS,省去计算时间，直接用ook
  %      begin=stop;
  %   end                  
  % 
     order(snr_iter)=256;
     SNR_dB=SNR(snr_iter);
     if   isinf(SNR(snr_iter)) 
            continue
     elseif SNR(1,snr_iter)<-15
        se_i=1;  
   SE=2+0.1*(se_i-1);
   lambda_NGMI=entropy2lambda(SE,TX.QAM.IQmap);  
   TX.SIG.lambda=lambda_NGMI; 
      lambda_snr(snr_iter)=lambda_NGMI;
    [gmi_snr(snr_iter),ngmi_snr(snr_iter)]=runonce(SNR_dB,nSpS,SIG,TX);%如果信噪比小于-15，查找表里没有，则和数组有，但SNR很小的情况一样，直接取se=2
     else 
        stop=unnamed(6,floor((SNR(snr_iter)-(-15))/0.5)+2);
        begin=unnamed(6,floor((SNR(snr_iter)-(-15))/0.5)+1);%由于SNR基本集中在-15到20，为了方便，一律设置成256qam

     TX.SIG.M= order(snr_iter);
TX.QAM = QAM_config(TX.SIG); %这里的config非常重要，如果没有这里重新config，上面根本改变不了阶数


  load order_NGMI_2D.mat
snri=round((SNR_dB-(-15))/0.5+1);%找到特定snr距离最近的索引
NGMI_order=order_NGMI_2D{1, log2(order)};
se_i=find(NGMI_order(snri,:)>=NGMI_thres, 1, 'last' );
if begin>stop   %这里为什么要换成正确的大小顺序，其实黄金分割算法无所谓区间始终点大小顺序，但是为了后面lambda_NGMI和区间的正确比较，还是要把begin和stop的值按正常大小排好
    a=stop;
    stop=begin;
    begin=a;
end

if ~isempty(se_i)
SE=2+0.1*(se_i-1);
lambda_NGMI=entropy2lambda(SE,TX.QAM.IQmap); 
  if lambda_NGMI>stop
      TX.SIG.lambda=lambda_NGMI; 
      lambda_snr(snr_iter)=lambda_NGMI;
    [gmi_snr(snr_iter),ngmi_snr(snr_iter)]=runonce(SNR_dB,nSpS,SIG,TX);%阈值非常小，直接取阈值点的λ，然后算对应gmi和ngmi
    
  elseif begin<=lambda_NGMI&&lambda_NGMI<=stop
          [gmi_snr(snr_iter),lambda_snr(snr_iter),ngmi_snr(snr_iter)]=cal_gmi_max(SNR_dB,nSpS,SIG,TX,begin,stop);%阈值在中间，则要算出最大点再来比较
          if lambda_NGMI>lambda_snr(snr_iter)
               TX.SIG.lambda=lambda_NGMI;
               lambda_snr(snr_iter)=lambda_NGMI;
               [gmi_snr(snr_iter),ngmi_snr(snr_iter)]=runonce(SNR_dB,nSpS,SIG,TX);%如果阈值还是比较小，还是要取阈值点，如果比较大，就不用算了，上面已经算出来最大点了
          end
  else 
      [gmi_snr(snr_iter),lambda_snr(snr_iter),ngmi_snr(snr_iter)]=cal_gmi_max(SNR_dB,nSpS,SIG,TX,begin,stop);%如果阈值很大，很容易达到，那可以直接搜索最大点
  end

else
   % [gmi_snr(snr_iter),lambda_snr(snr_iter),ngmi_snr(snr_iter)]=cal_gmi_max(SNR_dB,nSpS,SIG,TX,begin,stop);
   se_i=1;  
   SE=2+0.1*(se_i-1);
   lambda_NGMI=entropy2lambda(SE,TX.QAM.IQmap);  
   TX.SIG.lambda=lambda_NGMI; 
      lambda_snr(snr_iter)=lambda_NGMI;
    [gmi_snr(snr_iter),ngmi_snr(snr_iter)]=runonce(SNR_dB,nSpS,SIG,TX);
end

     end  

end%这里是snr结束

% save('power_alloc.mat','power_alloc');
% save('gmi_snr.mat','gmi_snr');
% save('lambda_snr.mat','lambda_snr');
% save('order.mat','order');
filename=['E:\NGMI_exp_data\NGMI_thre_0.92\str\str-entropy-loading\tx-lamda-mat\',num2str(t),'.mat'];
save(filename,"power_alloc","gmi_snr","lambda_snr","order","bit_alloc");

figure(1);
stem(SNR_sub);
ylabel('SNR average(dB)')
set(gca,'ytick',[-10:10:30])
figure();
stem(power_alloc);
ylabel('Power allocation(W)')

figure();
stem(bit_alloc);
ylabel('Bit allocation(bit)')
ylim([0 12])

end

