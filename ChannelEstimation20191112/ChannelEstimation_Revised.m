
%% Channel Estimation_for LS/DFT Channel Estimation with linear/Spline interpolation_Revised
%% Revised_Revised_Revised_Revised_Revised_Revised_Revised_Revised_Revised_Revised_Revised_Revised
close all;
clc
clf                                                                        % 清除当前图像窗口

%% Paramter Setting
Nfft = 32;                                                                 % FFT点数32
Ng = Nfft/8;                                                               % 循环前缀点数8
Nofdm = Nfft + Ng;                                                         % 一个OFDM符号总共有32+8=40点
Nsym = 100;                                                                % OFDM个数目为100个
Nps = 4;                                                                   % 导频间隔4
Np = Nfft/Nps;                                                             % 导频数8
Nd = Nfft-Np;                                                              % 数据点24 
Nbps = 4;
M = 2^Nbps;                                                                % 每个（已调制）符号的位数16

% mod_object = modem.qammod('M',M,'SymbolOrder','gray');
% 调制参数,modem.qammod已经被matlab高级版本删除,直接使用qammod调制即可
% demod_object = modem.qamdemod('M',M,'SymbolOrder','gray');              
% 解调参数,modem,qamdemod已经被matlab高级版本删除,直接使用deqammod解调即可

Es = 1;                                                                    % 信号能量
A = sqrt(3/2/(M-1)*Es);                                                    % QAM归一化因子 
SNRs = [30];                                                                  % 信噪比30dB
sq2 = sqrt(2);                                                             % 根号2

for i = 1:length(SNRs)
    SNR = SNRs(i);
    rand('seed',1);
    randn('seed',1);
    MSE = zeros(1,6);
    nose = 0;



















