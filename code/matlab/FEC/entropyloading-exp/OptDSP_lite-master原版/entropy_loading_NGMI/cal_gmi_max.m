function [gmi_snr,lambda_snr,ngmi_snr] = cal_gmi_max(SNR_dB,nSpS,SIG,TX,begin,stop)
a=begin;                            % start of interval
b=stop;                            % end of interval
epsilon=0.01;               % accuracy value
iter= 500;                       % maximum number of iterations
tau=double((sqrt(5)-1)/2);      % golden proportion coefficient, around 0.618
k=0;                            % number of iterations


x1=a+(1-tau)*(b-a);             % computing x values
x2=a+tau*(b-a);

TX.SIG.lambda=x1;
[f_x1,ng_x1]=runonce(SNR_dB,nSpS,SIG,TX);   
TX.SIG.lambda=x2;
[f_x2,ng_x2]=runonce(SNR_dB,nSpS,SIG,TX);

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
[f_x1,ng_x1]=runonce(SNR_dB,nSpS,SIG,TX);   

        
        % plot(x1,f_x1,'rx');
    else
        a=x1;
        x1=x2;
        x2=a+tau*(b-a);
        

f_x1=f_x2;   
TX.SIG.lambda=x2;
[f_x2,ng_x2]=runonce(SNR_dB,nSpS,SIG,TX);
        
        % plot(x2,f_x2,'rx')
    end
end
jingdu=abs(b-a);

% chooses minimum point
if(f_x1>f_x2)
    % sprintf('x_max=%f', x1)
    % sprintf('f(x_max)=%f ', f_x1)
    % plot(x1,f_x1,'ro')

   gmi_snr=f_x1;
   lambda_snr=x1;
   ngmi_snr=ng_x1;
else
    % sprintf('x_max=%f', x2)
    % sprintf('f(x_max)=%f ', f_x2)
    % plot(x2,f_x2,'ro')

    gmi_snr=f_x2;
   lambda_snr=x2;
   ngmi_snr=ng_x2;
end
end

