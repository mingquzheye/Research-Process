function [H_MMSE] = MMSE_CE(Y,Xp,pilot_loc,Nfft,Nps,h,SNR)
% MMSE Channel Estimation function
% Inputs:
%        Y         = Frequency_domain received signal
%        Xp        = Pilot signal
%        pliot_loc = Pilot Location
%        Nfft      = FFT size
%        Nps       = Pilot spacing
%        h         = Channel impulse response
%        SNR       = Signal-to-Noise Ratio[dB]
% Output:
%       H_MMSE     = MMSE channel estimation

snr = 10^(SNR*0.1);                                                        % 信噪比dB向真值转换
Np = Nfft/Nps;                                                             % 导频数

k = 1:Np;
H_tilde = Y(1,pilot_loc(k))./Xp(k);                                        % LS信道估计

k = 0:length(h)-1;                                                         % k_ts = k*ts;
hh = h*h';
tmp = h.*conj(h).*k;                                                       % tmp = h.*conj(h).*k_ts;

r = sum(tmp)/hh;
r2 = tmp*k.'/hh;                                                           % r2 = tmp*k_ts.'/hh;

tau_rms = sqrt(r2-r^2);                                                    % rms delay 到这里实际上是求最大平均时延tau_rms
df = 1/Nfft;                                                               % 1/(Nfft*ts)
j2pi_tau_df = 1j*2*pi*tau_rms*df;

K1 = repmat((0:Nfft-1).',1,Np);                                            % K1,sK2能看变量的值得到
K2 = repmat([0:Np-1],Nfft,1);

rf = 1./(1+j2pi_tau_df*(K1-K2*Nps));                                       %%?? 最大的问题在此，为什么需要使用(K1-K2*Nps)
%K1_K2chengNps = K1-K2*Nps;

K3 = repmat((0:Np-1).',1,Np);                                              % K3,K4能看变量的值得到
K4 = repmat([0:Np-1],Np,1);

rf2 = 1./(1+j2pi_tau_df*Nps*(K3-K4));                                      %%?? 问题在此，为什么需要使用(K3-K4)*Nps

Rhp = rf;                                                                  % 这里的互相关函数是真实信道h和LS估计信道序列p的相关，应该是32行8列
                                                                           % 32是真实信道序列长度，8是估计出的信道序列长度
Rpp = rf2+ eye(length(H_tilde),length(H_tilde))/snr;                       % 这里的自相关序列是LS估计信道序列p的自相关，应该是8行8列

% z = autocorr(H_tilde);
% Rpp_test = toeplitz(z);
% Rpp_test1 = corrmtx()
% 尝试用本方法来求相关函数Rpp，但是求出的结果并不能和上面的匹配

H_MMSE = transpose(Rhp*inv(Rpp)*H_tilde.');                                % 文中给出的求解公式

end