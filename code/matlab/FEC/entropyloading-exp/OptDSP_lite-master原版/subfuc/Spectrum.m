% function Spectrum(E,Fs)
% Npoints = length(E);
% FFT_E = fft(E,Npoints);
% FFT_E_origin = abs(FFT_E).*2/Npoints;
% Fre = (0:(Npoints-1))*Fs/Npoints;
% figure;
% plot(Fre./1e9, 10*log10(FFT_E_origin.^2),'r');
% % plot(Fre./1e9, FFT_E_origin,'r');
% % loglog(Fre./1e9, 10*log10(FFT_E_origin.^2),'r');
% xlabel('Frequency, GHz');
% ylabel('Power, dB');
% end



function Spectrum(E,Fs)
Npoints = length(E);
FFT_Ex_1 = fftshift(fft(E));
FFT_Ex = abs(FFT_Ex_1)./(length(E));
Frek = (Fs*(-(Npoints)/2:((Npoints/2)-1)))/Npoints;
figure;
plot(Frek./1e9, 10*log10(FFT_Ex.^2),'b');
% loglog(Frek./1e9, 10*log10(FFT_Ex.^2),'r');
% title('Spectrum of the received signal before the filter');
xlabel('Frequency(GHz)');
ylabel('Power(dB)');
end



