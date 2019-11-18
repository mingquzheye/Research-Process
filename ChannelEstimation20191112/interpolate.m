function [H_interpolated] = interpolate(H,pilot_loc,Nfft,method)
% Input: H           = Channel estimation using pilot sequence
%        pilot_loc   = Location of pilot sequence
%        Nfft        = FFT size
%        method      = 'linear'/'spline'
% Output:H_interpoltated = interpolated channel

if pilot_loc(1)>1                                                          % 为什么要有这两部分，是因为matlab中的所有插值法 interp1(x,y,xi,'method')
    slope = (H(2)-H(1))/(piloc_loc(2)-pilot_loc(1));                       % 要求x是单调的，并且xi不能超过x的范围
    H = [H(1)-slope*(pilot_loc(1)-1) H];                                   % 为了进行插值法，思路是使待插值估计矩阵必须首尾均有值才行
    pilot_loc = [1 pilot_loc];                                             % 这是为了保证首位（1）有值，参考内插公式
end

if pilot_loc(end)<Nfft                                                     % 这是为了保证尾位（Nfft）有值，参考内插共识                                 
    slope = (H(end)-H(end-1))/(pilot_loc(end)-pilot_loc(end-1));
    H = [H H(end)+slope*(Nfft-pilot_loc(end))];
    pilot_loc = [pilot_loc Nfft];
end

if lower(method(1))=='l'
    H_interpolated = interp1(pilot_loc,H,[1:Nfft]);
else
    H_interpolated = interp1(pilot_loc,H,[1:Nfft],'spline');
end

end