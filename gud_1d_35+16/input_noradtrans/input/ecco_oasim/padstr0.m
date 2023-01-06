function str=padstr0(N,len);
%padstr0--converts an int to a string and pads with zeros
%
%str=padstr0(N,len);
%
%

if(nargin<2)
    len=3;
end

str=int2str(N);
while(length(str)<len)
    str=['0',str];
end
if(length(str)>len)
    error(['integer N=',str,' has more than ',int2str(len),' digits']);
end