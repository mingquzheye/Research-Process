function output1=lmmse_estimation(input,pilot_inter,pilot_sequence,trms,t_max,snr)
%输入256行*120列的信号矩阵，导频间隔5，导频符号Xp是256行*1列，snr是信噪比的真值
%trms为多经信道的平均延时，此处所有的时间都是已经对采样间隔做了归一化后的结果，值为0.4937
%t_max为最大延时,此处所有的时间都是已经对采样间隔做了归一化后的结果，值为1.5744

beta=17/9;%----------------------------------------------------------------16QAM时系数是17/9，64QAM？？？？？？？？？？
[N,NL]=size(input);%-------------------------------------------------------输入信号矩阵的大小，N=256（行），NL=120（列）
Rhh=zeros(N,N);%-----------------------------------------------------------自相关矩阵为一个256行*256列的方阵
j=sqrt(-1);%---------------------------------------------------------------复数j

%%%%%%%%%%%%理想负指数分布时的自相关信道矩阵求解%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k=1:N
    for l=1:N
        Rhh(k,l)=(1-exp((-1)*t_max*((1/trms)+j*2*pi*(k-l)/N)))./(trms*(1-exp((-1)*t_max/trms))*((1/trms)+j*2*pi*(k-l)/N));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%计算LS信道估计%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i=1;
count=1;
while i<=NL
    Hi(:,count)=input(:,i)./pilot_sequence;
    count=count+1;
    i=i+pilot_inter+1;
end

%%%%%%根据公式Hlmmse=Rhh*Hls*inv(Rhh+beta/snr*I)计算LMMSE信道估计%%%%%%%%%%%%
weiner_coeff=Rhh*inv(Rhh+(beta/snr)*eye(N));
output1=weiner_coeff*Hi;

%   output2=weiner_coeff^2*Hi;%%%根据公式Hlmmse=Rhh*Hls*inv(Rhh+beta/snr*I)
%   output3=weiner_coeff^3*Hi;
%   output1(:,count)=weiner_coeff*Hi;%%%根据公式Hlmmse=Rhh*Hls*inv(Rhh+beta/snr*I)
%   output2(:,count)=weiner_coeff^2*Hi;%%%根据公式Hlmmse=Rhh*Hls*inv(Rhh+beta/snr*I)
%   output3(:,count)=weiner_coeff^3*Hi;
  
  
  
% while i<=NL
%     Hi=input(:,i)./pilot_sequence;
%     weiner_coeff=Rhh*inv(Rhh+(beta/snr)*eye(N));
%     H_LMMSE=weiner_coeff*Hi;%%%根据公式Hlmmse=Rhh*Hls*inv(Rhh+beta/snr*I)
%     %        Rx_symbols_dft=ifft( H_LMMSE);%%%直接做iFFT
%     %        Rx_symbols_ifft_dft=zeros(N,1);%%%先生成一个winnum*1的矩阵，
%     %        Rx_symbols_ifft_dft(1:winnum,:)=Rx_symbols_dft(1:winnum,:);% 只保留循环前缀长度CP=winnum以内的信道估计值，将循环前缀长度以为的信道估计值置为0
%     %        Rx_training_symbols_dft=fft(Rx_symbols_ifft_dft);
%    H_LMMSE_iteration1=weiner_coeff^2*Hi;%%%根据公式Hlmmse=Rhh*Hls*inv(Rhh+beta/snr*I)
%    H_LMMSE_iteration2=weiner_coeff^3*Hi;
% %     H_LMMSE_iteration1=Rhh*inv(Rhh+(beta/snr)*eye(N))*H_LMMSE;%%%根据公式Hlmmse=Rhh*Hls*inv(Rhh+beta/snr*I)
% %     H_LMMSE_iteration2=Rhh*inv(Rhh+(beta/snr)*eye(N))*H_LMMSE_iteration1;
%     output1(:,count)= H_LMMSE;
%     output2(:,count)= H_LMMSE_iteration1;
%     output3(:,count)= H_LMMSE_iteration2;
%     
%     
%     count=count+1;
%     i=i+pilot_inter+1;
% end

