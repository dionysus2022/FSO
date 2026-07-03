function [Sout,PS,A,B] = pulseShaper(Sin,nSpS,PS)

% Last Update: 31/03/2019


%% Input Parser
[nPol,nSamples] = size(Sin);

%% Select Pulse Shaping Filter
switch PS.type
    case {'Rect','rect','rectangular','none'}
        for n = 1:nPol
            Sout(n,:) = rectpulse(Sin(n,:),nSpS);
        end
    case {'RC','raised-cosine','raisedCos','Nyquist'}
        a = PS.rollOff;
        if isfield(PS,'nTaps')
            nTaps = PS.nTaps;
        else
            nTaps = 64 * nSpS;
        end
        k = -floor(nTaps/2):ceil(nTaps/2)-1;
        tK = k/nSpS;
        W = sinc(tK).*cos(a*pi*tK)./(1-4*a^2*tK.^2);
        W(isinf(W)) = 0;
              
        for n = 1:nPol
            Sout(n,:) = conv(upsample(Sin(n,:),nSpS),W,'same');
        end
        PS.W = W;
        
    case {'RRC','root-raised-cosine'}
        resampleFlag = abs(round(nSpS) - nSpS) > 1e-3;
        if resampleFlag
            nSpS_in = nSpS;
            nSpS = ceil(nSpS);
        end
        if isfield(PS,'nTaps')
            nTaps = PS.nTaps;
        else
            nTaps = 256*nSpS;
        end
        if isfield(PS,'implementation')
            implementation = PS.implementation;
        else
            implementation = 'FFT';
        end
        W = rcosdesign(PS.rollOff, nTaps/nSpS, nSpS, 'sqrt');   %W返回的是升余弦时域点
        W = W/sum(W);   %幅度上归一化                                                    % to guarantee unity gain at DC
        figure();
        plot(W);
        if strcmp(implementation,'FFT')
            zeroEnd = false;
            if mod(nSamples,2)
                Sin = [Sin zeros(nPol,1)];
                nSamples = nSamples + 1;
                zeroEnd = true;
            end
            W_f = [zeros(1,(nSpS*nSamples-nTaps)/2) W ...
                zeros(1,(nSpS*nSamples-nTaps)/2-1)];%三个数组拼接很多0，一个w再拼很多0，这里点数和后面上采样后的信号点数相同，才可以傅变相乘滤波
        end
        if strcmp(implementation,'conv')
            for n = 1:nPol
                Sout(n,:) = conv(upsample(Sin(n,:),nSpS),W,'same'); 
            end
        elseif strcmp(implementation,'FFT')
            for n = 1:nPol
                X = upsample(Sin(n,:),nSpS); %上采样，直接插零，nSpS为2则数值后插1个零，上采样本身不带来任何信息，只是起到压缩频谱的功能，降低了成型滤波器带宽要求，避免ISI，从时域上看，也是拉开了符号的距离，避免ISI
                % a=fftshift(fft(X));
                % figure();
                % plot(abs(a));
                % hold on
                A=W_f;
                B=X;
                Sout(n,:) = fftshift(ifft(fft(X).*fft(W_f)));%将输入信号和成型滤波器放到频谱相乘再反变换回来,shift是因为数组索引没有负数，所以matlab里的fft出来都是真正频谱周期延拓后取了0到taps,所以要将右边一半和左边一半交换位置，看上去才是真正频谱的样子
                % b=fftshift(fft(W_f));
                % figure();
                % plot(abs(b));
                % hold on
                % figure();
                % plot(abs(a.*b));
                % hold on
                % figure();
                % plot(abs(ifft(fft(X).*fft(W_f))));
                % hold on
                % a= conv(X,W_f);
                % stem(abs(a(8129:8149)));
                % hold on
                % b=ifft(fft(X).*fft(W_f));
                % stem(abs(b(1:20)));
            end
            if zeroEnd
                Sout = Sout(:,1:end-nSpS);
            end
        end
        if resampleFlag
            Sout = applyResample(Sout,nSpS,nSpS_in);
        end
        PS.W = W;
        
    case 'Gaussian'
        fcn = PS.fcn;
        nTaps = PS.nTaps;
        W = gaussdesign(fcn,nTaps/nSpS,nSpS);
        W = W/max(abs(W));
        for n = 1:nPol
            Sout(n,:) = conv(upsample(Sin(n,:),nSpS),W,'same');
        end
        PS.W = W;
                
    otherwise
        error('Invalid Pulse Shaping Filter!');
end

