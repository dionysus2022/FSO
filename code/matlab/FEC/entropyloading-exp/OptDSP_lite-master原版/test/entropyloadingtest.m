function [power_alloc,gmi_snr,lambda_snr,order] = entropyloadingtest(NumberOfCarriers,ch_fading,nSpS,SIG,TX,t)
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
for snr_iter = 1:N_subc
   if   isinf(SNR(snr_iter)) 
        continue
  elseif (17<SNR(snr_iter))
        order(snr_iter)=1024;
        stop=unnamed(8,floor((SNR(snr_iter)-(-15))/0.5)+2);%由于lambda查找表从-15到35db，所以算距离-15差多少，然后除以0.5这个步长，再加上1这个数组初始索引。
        begin=unnamed(8,floor((SNR(snr_iter)-(-15))/0.5)+1);
  elseif (11<SNR(1,snr_iter))&&(SNR(snr_iter)<=17 )
        order(snr_iter)=256;
        stop=unnamed(6,floor((SNR(snr_iter)-(-15))/0.5)+2);
        begin=unnamed(6,floor((SNR(snr_iter)-(-15))/0.5)+1);
  elseif (5<SNR(snr_iter))&&(SNR(snr_iter)<=11 )
        order(snr_iter)=64;
        stop=unnamed(4,floor((SNR(snr_iter)-(-15))/0.5)+2);
        begin=unnamed(4,floor((SNR(snr_iter)-(-15))/0.5)+1);
  elseif  (-15<=SNR(snr_iter))&&(SNR(snr_iter)<=5 )
       order(snr_iter)=16;
        stop=unnamed(2,floor((SNR(snr_iter)-(-15))/0.5)+2);
        begin=unnamed(2,floor((SNR(snr_iter)-(-15))/0.5)+1);
   elseif SNR(1,snr_iter)<-15
       order(snr_iter)=2;
       stop=1.6;%如果信噪比小于-15，则不进行PS,省去计算时间，直接用ook
       begin=stop;
    end
     TX.SIG.M= order(snr_iter);
TX.QAM = QAM_config(TX.SIG); %这里的config非常重要，如果没有这里重新config，上面根本改变不了阶数
a=begin;                            % start of interval
b=stop;                            % end of interval
epsilon=0.005;               % accuracy value  调节以上三者可以用来加速
iter= 500;                       % maximum number of iterations
tau=double((sqrt(5)-1)/2);      % golden proportion coefficient, around 0.618
k=0;                            % number of iterations


x1=a+(1-tau)*(b-a);             % computing x values
x2=a+tau*(b-a);

TX.SIG.lambda=x1;

f_x1=runonce(SNR(1,snr_iter),nSpS,SIG,TX);   
TX.SIG.lambda=x2;
f_x2=runonce(SNR(1,snr_iter),nSpS,SIG,TX);

% plot(x1,f_x1,'rx')              % plotting x
% plot(x2,f_x2,'rx')

while ((abs(b-a)>epsilon) && (k<iter))
    k=k+1;
    if(f_x1>f_x2)
        b=x2;
        x2=x1;
        x1=a+(1-tau)*(b-a);

f_x2= f_x1;       
TX.SIG.lambda=x1;
f_x1=runonce(SNR(1,snr_iter),nSpS,SIG,TX);   

        
        % plot(x1,f_x1,'rx');
    else
        a=x1;
        x1=x2;
        x2=a+tau*(b-a);
        

f_x1=f_x2;   
TX.SIG.lambda=x2;
f_x2=runonce(SNR(1,snr_iter),nSpS,SIG,TX);
        
        % plot(x2,f_x2,'rx')
    end
end
jingdu=abs(b-a);


if(f_x1>f_x2)
    % sprintf('x_max=%f', x1)
    % sprintf('f(x_max)=%f ', f_x1)
    % plot(x1,f_x1,'ro')

   gmi_snr(snr_iter)=f_x1;
   lambda_snr(snr_iter)=x1;
else
    % sprintf('x_max=%f', x2)
    % sprintf('f(x_max)=%f ', f_x2)
    % plot(x2,f_x2,'ro')

    gmi_snr(snr_iter)=f_x2;
      lambda_snr(snr_iter)=x2;
end
end

% save('power_alloc.mat','power_alloc');
% save('gmi_snr.mat','gmi_snr');
% save('lambda_snr.mat','lambda_snr');
% save('order.mat','order');
filename=['C:\Users\LZY\Desktop\entropyloading-exp\OptDSP_lite-master原版\test\',num2str(t),'lambda.mat'];
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

