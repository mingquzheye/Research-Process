clear all
close all
clc
format long %--------------------------------------------------------------表示15位定点近似数

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%参数预设%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pilot_inter=5;%------------------------------------------------------------导频间隔5
N_carrier=256;%------------------------------------------------------------子载波数256个
channel_type = 'EPA';%-----------------------------------------------------认为是在下面定义的EPA信道里面进行信道估计的
j=sqrt(-1);%---------------------------------------------------------------就表示虚数i
cp_length=16;%-------------------------------------------------------------循环前缀长度16
SNR_dB=1:2:26;%------------------------------------------------------------信噪比依次可以取值，15、20和25dB
modemNum =16;%-------------------------------------------------------------64QAM
loop_num=1;%---------------------------------------------------------------循环数一次
ofdm_symbol_num=100;%------------------------------------------------------OFDM符号数100个
sample_rate = 3.84e6;%-----------------------------------------------------采样率3840000
t_interval = 1/sample_rate;%-----------------------------------------------采样时间间隔，是采样率的倒数；为离散信道抽样时间间隔，等于OFDM符号长度/(子载波个数+cp长度lp)
fd=10;%--------------------------------------------------------------------多普勒频移10Hz
counter=200000;%-----------------------------------------------------------计数器200000

if(channel_type ==  'EPA')%------------------------------------------------EPA信道，7条路径，时延值已经给出
    delay = ([0 30 70 90 110 190 410])*1e-9;
    pathNum = 7;
elseif(channel_type ==  'EVA')%--------------------------------------------EVA信道，9条路径，时延值已经给出
    delay =  ([0 30 150 310 370 710 1090 1730 2510])*1e-9;
    pathNum = 9;
elseif(channel_type ==  'ETU')%--------------------------------------------ETUI信道，9条路径，时延值已经给出
    delay =  ([0 50 120 200 230 500 1600 2300 5000])*1e-9;
    pathNum = 9;
else
    error('Channel mode WRONG!!!');
end

trms = mean(delay);%-------------------------------------------------------取平均时延值
var_pow=10*log10(exp(-delay/trms));%---------------------------------------各径相对主径的平均功率,单位dB
% trms_1=trms;
% t_max=max(delay);
trms_1=trms/t_interval;%---------------------------------------------------trms为多经信道的平均延时，此处所有的时间都是已经对采样间隔做了归一化后的结果
t_max=max(delay)/t_interval;%----------------------------------------------%t_max为最大延时,此处所有的时间都是已经对采样间隔做了归一化后的结果

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%仿真开始%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length(SNR_dB)
    
    LS_error_num = 0;%-----------------------------------------------------LS错误数
    DFT_error_num = 0;%----------------------------------------------------
    LMMSE_error_num = 0;%--------------------------------------------------
    SVD_error_num = 0;%----------------------------------------------------
    MMSE_error_num = 0;%---------------------------------------------------
    
    for l=1:loop_num%------------------------------------------------------循环一次
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%生成导频符号%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        pilot_symbol = 2*randi([0,1])-1  + (2*randi([0,1])-1)*j;%----------导频只有一个，导频长什么样：-1+1j
        bit_source=randi([0 1],N_carrier*ofdm_symbol_num*log2(modemNum),1);%产生0或者1的随机分布，共计N_carrier*ofdm_symbol_num*log2(modemNum)=153600行*1列
        nbit=length(bit_source);%------------------------------------------需要调制的bit数据的长度
       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%64QAM调制%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %h = modem.pskmod('M', modemNum, 'SymbolOrder', 'Gray', 'InputType', 'bit');
        h = modem.qammod('M', modemNum, 'SymbolOrder', 'Gray', 'InputType', 'bit');
        %------------------------------------------------------------------定义一个调制系统：定义64QAM，使用格雷码，输入数据类型是bit       
        map_out_pre = modulate(h, bit_source);
        %------------------------------------------------------------------64QAM调制       
        map_out = reshape(map_out_pre,N_carrier,ofdm_symbol_num);
        %------------------------------------------------------------------经过reshape，将得到的调制序列变成 N_carrier行*fdm_symbol_num列，即256行*100列的调制符号矩阵，这里即满足100个信号符号和256个载频 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%插入导频符号%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        [insert_pilot_out,pilot_num,pilot_sequence]=insert_pilot(pilot_inter,pilot_symbol,map_out);
        %------------------------------------------------------------------调用导频插入函数，输入导频间隔，导频数据（导频长什么样），以及得到的256行*100列的已调信号
        %------------------------------------------------------------------输出结果分三个部分：插入导频后的序列、插入导频的个数、插入导频数据（插入导频符号长什么样）
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%IFFT模块%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        ofdm_modulation_out=ifft(insert_pilot_out,N_carrier);%-------------对导入导频后的序列进行256点FFT逆变换，变换到时域
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%插入循环保护间隔%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
        
        ofdm_cp_out=insert_cp(ofdm_modulation_out,cp_length);
        %------------------------------------------------------------------调用循环前缀插入函数，输入时域信号和循环前缀长度
              
        %counter=200000;%各径信道的采样点间隔，应该大于信道采样点数。由以上条件现在信道采样点数
        %count_begin=(l-1)*(5*counter);%每次仿真信道采样的开始位置
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%并串转换+过信道%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %passchan_ofdm_symbol=multipath_chann(ofdm_cp_out,pathNum,var_pow,delay,fd,t_interval,counter,count_begin);
        %得到7行*32640列的通过信道之后的7条路径的响应（经受了频率衰减性）
        %上面的思路有问题：因为根据OFDM的原理图，应该要先将信号序列变成串行数据，然后才通过信道，而上面的编程并没有并串转换
        
        serial_ofdm_cp_out = reshape(ofdm_cp_out,272*120,1);%--------------这里做个就将插入循环前缀后的信号变成串行的数据
         [Ts, Fd, Tau, Pdb]=LTEchannel(channel_type);%---------------------由EPA信道的特性得到相应的参数
         Fading_chan= rayleighchan(Ts, Fd, Tau, Pdb);%---------------------由上面得到的参数建立瑞利信道模型
         passchan_ofdm_symbol = filter(Fading_chan,serial_ofdm_cp_out);%---对串行数据用瑞利信道模型对其进行滤波，然后就得到通过信道的数据序列
         passchan_nonoise_matrix = reshape(passchan_ofdm_symbol,272,120);%-将通过信道衰减后的符号再次变成272行*120列的信号矩阵
        
%%%%%%%%%%%%%%%%%%%%%%%%%ofdm符号加高斯白噪声%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        receive_ofdm_symbol=awgn(passchan_ofdm_symbol,SNR_dB(i),'measured'); %将经过信道衰减的信号序列再附加上高斯白噪声
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%去除循环前缀%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        passchan_withnoise_parallel = reshape(receive_ofdm_symbol,272,120);%-附加高斯白噪声之后，接下来将串行数据再次转换为并行数据
       
        cutcp_ofdm_symbol_nonoise=cut_cp(passchan_nonoise_matrix,cp_length);
        %------------------------------------------------------------------去除循环前缀,输入信号是272行*120列，没有附加高斯白噪声
        cutcp_ofdm_symbol=cut_cp(passchan_withnoise_parallel,cp_length);
        %------------------------------------------------------------------去除循环前缀，输入信号是272行*120列，附加了高斯白噪声
        
        var(i)=sum(sum(abs(cutcp_ofdm_symbol-cutcp_ofdm_symbol_nonoise).^2))/(N_carrier*ofdm_symbol_num);  
        %------------------------------------------------------------------算的是噪声的方差

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%FFT模块%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        ofdm_demodulation_out_nonoise=fft(cutcp_ofdm_symbol_nonoise,N_carrier);
        %------------------------------------------------------------------FFT模块、输入信号是256行*120列，没有附加高斯白噪声
        ofdm_demodulation_out=fft(cutcp_ofdm_symbol,N_carrier);
        %------------------------------------------------------------------FFT模块、输入信号是256行*120列，附加了高斯白噪声
        
%%%%%%%%%%%%%重点：以下就是对接收OFDM信号进行信道估计和信号检测的过程%%%%%%%%%%
        snr=10^(SNR_dB(i)/10);%----------------------------------------------------------------------------------将信噪比的dB值变为真值
        H_real=ls_estimation(ofdm_demodulation_out_nonoise,pilot_inter,pilot_sequence);%-------------------------通过FFT后没有附加高斯白噪声的信号视为信道序列测量的真实值
        H_LS=ls_estimation(ofdm_demodulation_out,pilot_inter,pilot_sequence);%-----------------------------------LS信道估计
        H_DFT=ls_DFT_estimation(ofdm_demodulation_out,pilot_inter,pilot_sequence,cp_length);%--------------------LS_DFT信道估计
        H_LMMSE=lmmse_estimation(ofdm_demodulation_out,pilot_inter,pilot_sequence,trms_1,t_max,snr);%------------LMMSE信道估计
        H_SVD=lr_lmmse_estimation(ofdm_demodulation_out,pilot_inter,pilot_sequence,trms_1,t_max,snr,cp_length);%-SVD信道估计
        H_MMSE=mmse_estimation(ofdm_demodulation_out,pilot_inter,pilot_sequence,trms_1,t_max,var(i));%-----------MMSE信道估计
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%统计MSE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        MSE_LS(l)=sum(sum(abs(H_LS-H_real).^2))/(N_carrier*ofdm_symbol_num/pilot_inter);%------------------------计算平方误差，估计信道与真实信道每个元素之间的差值求平方和
        MSE_DFT(l)=sum(sum(abs(H_DFT-H_real).^2))/(N_carrier*ofdm_symbol_num/pilot_inter);%----------------------这个H_real是真正的real吗
        MSE_LMMSE(l)=sum(sum(abs(H_LMMSE-H_real).^2))/(N_carrier*ofdm_symbol_num/pilot_inter);
        MSE_SVD_MMSE(l)=sum(sum(abs(H_SVD-H_real).^2))/(N_carrier*ofdm_symbol_num/pilot_inter);
        MSE_MMSE(l)=sum(sum(abs(H_MMSE-H_real).^2))/(N_carrier*ofdm_symbol_num/pilot_inter);
        
     
%%%%%%%%%%%%%%%%%%%%%%%%%%时域线性插值与数据恢复%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        LS_data  = linear_interpolation(ofdm_demodulation_out,H_LS,pilot_inter);%----------------------------------时域插值？？？OFDM符号不是100个吗，这里怎么变成了95个？
        DFT_data  = linear_interpolation(ofdm_demodulation_out,H_DFT,pilot_inter);
        LMMSE_data  = linear_interpolation(ofdm_demodulation_out,H_LMMSE,pilot_inter);
        SVD_data  = linear_interpolation(ofdm_demodulation_out,H_SVD,pilot_inter);
        MMSE_data  = linear_interpolation(ofdm_demodulation_out,H_MMSE,pilot_inter);
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 解调 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %H_demod = modem.pskdemod('M', modemNum, 'SymbolOrder', 'Gray', 'OutputType', 'bit');%
        H_demod = modem.qamdemod('M', modemNum, 'SymbolOrder', 'Gray', 'OutputType', 'bit');%----------------------信号解调
        LS_demod_bit = demodulate(H_demod,LS_data);%---------------------------------------------------------------OFDM符号不是100个吗，这里怎么也变成了95个？
        DFT_demod_bit = demodulate(H_demod,DFT_data);
        LMMSE_demod_bit = demodulate(H_demod,LMMSE_data);
        SVD_demod_bit = demodulate(H_demod,SVD_data);
        MMSE_demod_bit = demodulate(H_demod,MMSE_data);
        
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 统计错误bit %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for m = 1:ofdm_symbol_num - pilot_inter%---------------------------只统计了95个OFDM里面的BER误差
            for n = 1:N_carrier*log2(modemNum);
                if (LS_demod_bit(n,m) ~= bit_source((m-1)*N_carrier*log2(modemNum) + n))
                    LS_error_num = LS_error_num +1;
                end
                if (DFT_demod_bit(n,m) ~= bit_source((m-1)*N_carrier*log2(modemNum) + n))
                    DFT_error_num = DFT_error_num +1;
                end
                if (LMMSE_demod_bit(n,m) ~= bit_source((m-1)*N_carrier*log2(modemNum) + n))
                    LMMSE_error_num = LMMSE_error_num +1;
                end
                if (SVD_demod_bit(n,m) ~= bit_source((m-1)*N_carrier*log2(modemNum) + n))
                    SVD_error_num = SVD_error_num +1;
                end
                if (MMSE_demod_bit(n,m) ~= bit_source((m-1)*N_carrier*log2(modemNum) + n))
                    MMSE_error_num = MMSE_error_num +1;
                end
            end
        end
        
%    LS_BER(l) = LS_error_num / ((ofdm_symbol_num - pilot_inter)*N_carrier*log2(modemNum));
%    DFT_BER(l) = DFT_error_num / ((ofdm_symbol_num - pilot_inter)*N_carrier*log2(modemNum));
%    MMSE_BER(l) = MMSE_error_num / ((ofdm_symbol_num - pilot_inter)*N_carrier*log2(modemNum));
%    LMMSE_BER(l) = LMMSE_error_num / ((ofdm_symbol_num - pilot_inter)*N_carrier*log2(modemNum));
%    SVD_BER(l) = SVD_error_num / ((ofdm_symbol_num - pilot_inter)*N_carrier*log2(modemNum));
        
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MSE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    MSE_LS_dB(i)=mean(MSE_LS);%--------------------------------------------对误差的平方求平均，这里才是均方误差
    MSE_DFT_dB(i)=mean(MSE_DFT);
    MSE_MMSE_dB(i)=mean(MSE_MMSE);
    MSE_LMMSE_dB(i)=mean(MSE_LMMSE);
    MSE_SVD_MMSE_dB(i)=mean(MSE_SVD_MMSE);
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%% BER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Num_bit= ((ofdm_symbol_num - pilot_inter)*N_carrier*log2(modemNum));%--算误码率
    LS_BER_dB(i) = LS_error_num /Num_bit;
    DFT_BER_dB(i) = DFT_error_num / Num_bit;
    MMSE_BER_dB(i) = MMSE_error_num / Num_bit;
    LMMSE_BER_dB(i) = LMMSE_error_num / Num_bit;
    SVD_BER_dB(i) = SVD_error_num /Num_bit;
    
end

figure(1);
semilogy(SNR_dB,MSE_LS_dB,'-ro',SNR_dB,MSE_DFT_dB,'-black^',SNR_dB,MSE_MMSE_dB,'-bo',SNR_dB,MSE_LMMSE_dB,'-c+',SNR_dB,MSE_SVD_MMSE_dB,'--md','LineWidth',1.5);
legend('LS','IDFT-DFT','MMSE','LMMSE','SVD-MMSE');
grid on
xlabel('SNR-dB');
ylabel('MSE');
title('信道估计算法MSE性能比较');

% saveas(gcf, 'C:\Users\Administrator\Desktop\Simulation_diagram\信道估计EPA_Fd10_MSE_64QAM_256_20', 'fig');

figure(2);
semilogy(SNR_dB,LS_BER_dB,'-ro',SNR_dB,DFT_BER_dB,'-black^',SNR_dB,MMSE_BER_dB,'-bo',SNR_dB,LMMSE_BER_dB,'-c+',SNR_dB,SVD_BER_dB,'--md','LineWidth',1.5);
legend('LS','IDFT-DFT','MMSE','LMMSE','SVD-MMSE');
grid on
xlabel('SNR-dB');
ylabel('BER');
title('信道估计算法BER性能比较');

% saveas(gcf,  'C:\Users\Administrator\Desktop\Simulation_diagram\信道估计EPA_Fd10_BER_64QAM_256_20', 'fig');
