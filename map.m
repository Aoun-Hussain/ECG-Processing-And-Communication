function output = map(data,min,max,minOutput,maxOutput)
%MAP Summary of this function goes here
%   Detailed explanation goes here
for i=1:size(data,2)
   output(i)=(data(i)-min)*(maxOutput-minOutput)/(max-min)+minOutput;
end
end

