function [output,count,pilot_sequence]=insert_pilot(pilot_inter,pilot_symbol,map_out_block)  % 5个块状导频（OFDM符号）、大小：1*1、已调信号：256行*100列
% pilot_inter为导频符号间隔，以ofdm符号个数的形式给出，为整数，间隔越大估计误差越大
% pilot_symbol_bit 采用常数导频符号，这里给出它的二进制形式
% map_out_block 映射后的一次仿真所包含的符号块
  [N,NL]=size(map_out_block);%---------------------------------------------得到已调信号矩阵的行数和列数：行代表的是子载波，行数为256；列代表的是OFDM符号，列数为100
  output=zeros(N,(NL+fix(NL/pilot_inter))); %------------------------------得到插入导频后的输出矩阵output是256行*120列
  pilot_sequence=pilot_symbol*ones(N,1);%----------------------------------得到块状导频（每个块状导频相当于一个OFDM符号），为256行*1列，里面的值都为-1+1i
  
  count=0;%----------------------------------------------------------------记录插入导频信号的次数,一共要将20个导频全部插入到已调信号中去
  i=1;
  while i<(NL+fix(NL/pilot_inter))%----------------------------------------每隔pilot_inter个符号(5个)插入一个导频序列，i>120列时停止循环
      output(:,i)=pilot_sequence;%-----------------------------------------将导频符号赋值给Output的某些列，每隔5个OFDM符号插入一个导频符号，插入导频
      count=count+1;%------------------------------------------------------然后完成一次插值
   if count*pilot_inter<=NL%-----------------------------------------------在20个导频符号插完之前，对调制后的OFDM符号寻找插值都为列位置
      output(:,(i+1):(i+pilot_inter))=map_out_block(:,((count-1)*pilot_inter+1):count*pilot_inter);
   else%-------------------------------------------------------------------在20个导频符号插完之后，将剩余的OFDM符号，全部放在output的最后列里面去
      output(:,(i+1):(i+pilot_inter+NL-count*pilot_inter))=map_out_block(:,((count-1)*pilot_inter+1):NL);
   end
      i=i+pilot_inter+1;
  end

          