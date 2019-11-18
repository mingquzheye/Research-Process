function [H_LS] = LS_CE(Y,Xp,pilot_loc,Nfft,Nps,int_opt)
% LS Estimation
% Inputs:
%        Y         = Frequency_domain received signal
%        Xp        = Pilot signal
%        pliot_loc = Pilot Location
%        N         = FFT size
%        Nps       = Pilot spacing
%        int_opt   = 'linear' or 'spline'
% Output:
%        H_LS      = LS Channel Estimation

Np = Nfft/Nps;                                                             % 导频数
k = 1:Np;                                                                  % 便利所有的导频位置
LS_est(k) = Y(pilot_loc(k))./Xp(k);                                        % LS信道估计（导频位置处的信道估计）

if lower(int_opt(1)=='l')                                                  % 判断插值方法
    method = 'linear';
else
    method = 'spline';
end

H_LS = interpolate(LS_est,pilot_loc,Nfft,method);                          % 插值,这里的interpolate是自己定义的函数

end
