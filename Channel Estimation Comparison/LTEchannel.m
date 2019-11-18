function [Ts, Fd, Tau, Pdb]=LTEchannel(channel_type)%%%信道模型参考论文：LTE信道估计以及干扰噪声测量

Ts = 1e-6/3.84;%the sample time of the input signal-------------30.72MHZ-

% ITU Channel Models
if strcmp(channel_type,'EPA')==1   %Extended Pedestrian A model 步行A模型7径
    Fd = 5; % FD is the maximum Doppler shift, in Hertz;
    Pdb = [0 -1.0 -2.0 -3.0 -8.0 -17.2 -20.8];
    Tau = ([0 30 70 90 110 190 410])*1e-9;  
elseif strcmp(channel_type,'EVA')==1 %Extended Vehicular A model  车载A型9径
    Fd = 600;% FD is the maximum Doppler shift, in Hertz;
    Pdb = [0 -1.5 -1.4 -3.6 -0.6 -9.1 -7.0 -12.0 -16.9];
    Tau = ([0 30 150 310 370 710 1090 1730 2510])*1e-9;
    
elseif strcmp(channel_type,'ETU')==1 %Extended Typical Urban model  城市模型9径
    Fd = 556;% FD is the maximum Doppler shift, in Hertz;
    Pdb = [-1.0 -1.0 -1.0 0 0 0 -3.0 -5.0 -7.0];
    Tau = ([0 50 120 200 230 500 1600 2300 5000])*1e-9;
elseif strcmp(channel_type,'TU6')==1%%%%%室内环境6径
    Fd = 0.1;
    Pdb = [-3.0  0  -2.0  -6.0  -8.0  -10.0];
    Tau = ([0 0.2 0.5 1.6 2.3 5.0])*1e-6;
    
else
    error('Channel mode wrong input.');
end



