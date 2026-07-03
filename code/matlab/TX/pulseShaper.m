%离散符号→连续脉冲：通过上采样 + 滤波，把离散符号转成带特定形状的连续信号，避免码间串扰、控制带宽；
%根据指定的 "滤波器类型"（矩形、升余弦、根升余弦、高斯），对输入的离散符号信号Sin做脉冲成型，输出连续的脉冲信号Sout。
function [Sout,PS] = pulseShaper(Sin,nSpS,PS)

% Last Update: 31/03/2019


%% Input Parser
[nPol,nSamples] = size(Sin);  % 获取输入信号的维度：nPol=偏振数，nSamples=符号数

%% Select Pulse Shaping Filter  核心逻辑 —— 根据滤波器类型执行不同成型策略
switch PS.type  %进行分支判断，对中常用的滤波器应
    case {'Rect','rect','rectangular','none'} %矩形滤波器
        for n = 1:nPol   % 逐偏振处理
            Sout(n,:) = rectpulse(Sin(n,:),nSpS);  % 矩形脉冲成型
        end

    case {'RC','raised-cosine','raisedCos','Nyquist'}  %升余弦滤波器（RC，Nyquist 滤波器，无码间串扰）
        a = PS.rollOff;  % 滚降系数（0~1，越小带宽越窄，成型效果越"尖锐"）
         % 滤波器抽头数：用户指定或默认64×nSpS（抽头数越多，滤波效果越精准）
        if isfield(PS,'nTaps')
            nTaps = PS.nTaps;
        else
            nTaps = 64 * nSpS;
        end

         % 生成滤波器系数的横坐标（归一化时间）
        k = -floor(nTaps/2):ceil(nTaps/2)-1;   % 抽头索引（对称，比如-32到31）
        tK = k/nSpS;   % 归一化到"符号周期"（1个符号周期=1）
        % 升余弦滤波器的核心公式（不用记，知道作用即可）
        W = sinc(tK).*cos(a*pi*tK)./(1-4*a^2*tK.^2);
        W(isinf(W)) = 0;  % 升余弦滤波器的核心公式（不用记，知道作用即可）

        % 逐偏振滤波：上采样+卷积
        for n = 1:nPol
            % 步骤1：上采样（插零）——把离散符号变成"1个符号对应nSpS个点，除了原符号位置都是0"
            % 例子：[1,-1]，nSpS=2 → [1,0,-1,0]
            % 步骤2：卷积（conv）——用升余弦滤波器系数W"平滑"插零后的信号，得到连续脉冲
            % 'same'：输出和输入长度一致
            Sout(n,:) = conv(upsample(Sin(n,:),nSpS),W,'same');
        end
        PS.W = W; % 保存滤波器系数
        

    case {'RRC','root-raised-cosine'} %根升余弦滤波器（RRC，实际系统最常用）
         % 第一步：处理非整数nSpS（上采样点数）的情况
        resampleFlag = abs(round(nSpS) - nSpS) > 1e-3;
        if resampleFlag
            nSpS_in = nSpS;
            nSpS = ceil(nSpS);   % 向上取整（比如3.2→4）
        end
         % 第二步：滤波器抽头数（用户指定或默认256×nSpS，比RC更多，更精准）
        if isfield(PS,'nTaps')
            nTaps = PS.nTaps;
        else
            nTaps = 256*nSpS;
        end
        % 第三步：滤波实现方式（FFT/卷积，默认FFT——速度更快）
        if isfield(PS,'implementation')
            implementation = PS.implementation;
        else
            implementation = 'FFT';
        end
        % 第四步：生成根升余弦滤波器系数（Matlab内置函数，比手动算RC更方便）
        W = rcosdesign(PS.rollOff, nTaps/nSpS, nSpS, 'sqrt');
        W = W/sum(W);   %幅度上归一化（保证直流增益为1，信号功率不变）
        % 
        % to guarantee unity gain at DC

         % 分支A：FFT实现（频域卷积，速度快，适合大数据）
        if strcmp(implementation,'FFT')
            zeroEnd = false;
            % 保证符号数为偶数（FFT要求长度为2的幂/偶数，避免频谱失真）
            if mod(nSamples,2)
                Sin = [Sin zeros(nPol,1)];% 补零
                nSamples = nSamples + 1;
                zeroEnd = true;
            end
             % 构造和上采样后信号长度一致的滤波器系数（补零）
            W_f = [zeros(1,(nSpS*nSamples-nTaps)/2) W ...
                zeros(1,(nSpS*nSamples-nTaps)/2-1)];
        end

          % 分支B：卷积实现（时域卷积，直观，适合小数据）
        if strcmp(implementation,'conv')
            for n = 1:nPol
                Sout(n,:) = conv(upsample(Sin(n,:),nSpS),W,'same'); 
            end
             % 分支A续：FFT实现的核心（频域相乘=时域卷积，速度快）
        elseif strcmp(implementation,'FFT')
            for n = 1:nPol
                X = upsample(Sin(n,:),nSpS); %上采样，直接插零，nSpS为2则数值后插1个零
                Sout(n,:) = fftshift(ifft(fft(X).*fft(W_f)));%将输入信号和成型滤波器放到频谱相乘再反变换回来
            end
            if zeroEnd
                Sout = Sout(:,1:end-nSpS);% 去掉之前补的零
            end
        end

        % 处理非整数nSpS的情况（重采样回原采样率）
        if resampleFlag
            Sout = applyResample(Sout,nSpS,nSpS_in);
        end
        PS.W = W; % 保存滤波器系数
        
    case 'Gaussian' %高斯滤波器（光通信 / 短波通信常用）
        fcn = PS.fcn;   % 高斯滤波器的3dB带宽符号周期积（成型因子）
        nTaps = PS.nTaps;  % 滤波器抽头数（用户必须指定）
        % 生成高斯滤波器系数（Matlab内置函数）
        W = gaussdesign(fcn,nTaps/nSpS,nSpS);
        W = W/max(abs(W)); % 幅度归一化（峰值为1）
        % 时域卷积实现成型
        for n = 1:nPol
            Sout(n,:) = conv(upsample(Sin(n,:),nSpS),W,'same');
        end
        PS.W = W;
                
    otherwise
        error('Invalid Pulse Shaping Filter!');
end

