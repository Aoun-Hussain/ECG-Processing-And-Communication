if exist('a','var') && isa(a,'arduino') && isvalid(a)
    % nothing to do
else
    a = arduino('COM4','Due');
end


