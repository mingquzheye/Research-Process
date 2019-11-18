function output = linear_interpolation(received_data,H_estimation,pilot_inter)
%输入的是256行*126列经过解调之后的信号矩阵（这里我认为有问题，应该先将那20个导频符号先取出来，然后才能进行解调）
%输入经过不同方式进行信道估计之后的导频位置处的信道响应值
%导频间隔个5


[Ncarriers,NL]=size(received_data);%---------------------------------------接收到的信号矩阵的大小，为Ncarrier=256（行），NL=120（列）
[NCarriers,Npilot]=size(H_estimation);%------------------------------------得到导频位置处的信道响应估计值矩阵大小，为Ncarrier=256（行），Npilot=20（列）


H_data_plus_pilot = zeros(Ncarriers,NL-pilot_inter);%----------------------去除导频之后的解调信号矩阵大小，为Ncarrier=256（行），NL-pilot_inter=115(列)？？？？？？？

for j = 1:Npilot-1
    for i=1:Ncarriers
        H_data_plus_pilot(i, (pilot_inter+1)*(j-1)+1  :  (pilot_inter+1)*(j-1)+(pilot_inter+2) ) = linspace( H_estimation(i, j),   H_estimation(i, j+1), (pilot_inter+2));
    end
end


for i = 1:NL-pilot_inter
    recover_data(:, i ) = received_data(:, i)./H_data_plus_pilot(:,i );
end

OFDM_symbols = NL-Npilot-pilot_inter;

output = zeros(Ncarriers,OFDM_symbols);

for i = 1 : OFDM_symbols
    output(:, i) = recover_data(:, i+ceil(i/pilot_inter));
end

%？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？对该做法表示怀疑，最好采用之前用过的interp1函数


