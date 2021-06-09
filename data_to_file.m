%ecg=importdata('asd.txt');
ecg=importdata('Noise.txt');
%test1=importdata('Orig.txt');
scaledECG=uint16(map(ecg,min(ecg),max(ecg),0,4095));
str = num2str(scaledECG);
str = regexprep(str,'\s+',',');
compSize=num2str(size(scaledECG,2));
% Save as 'sample.h' file
fid = fopen('wave.h','w');
fprintf(fid,"#include <avr/pgmspace.h>\n ");
fprintf(fid,'const PROGMEM uint16_t waveTable[] = {%s};\n',str);
fclose(fid);
