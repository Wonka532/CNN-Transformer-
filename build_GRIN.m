function [epsilon, x, dx] = build_GRIN(lambda, Nx, spatial_window, radius, extra_params)

a1=0.6961663;
a2=0.4079426;
a3=0.8974794;
b1= 0.0684043;
b2=0.1162414;
b3=9.896161;

nsi=(1+a1*(lambda.^2)./(lambda.^2 - b1^2)+a2*(lambda.^2)./(lambda.^2 - b2^2)+a3*(lambda.^2)./(lambda.^2 - b3^2)).^(0.5);

nco = nsi + extra_params.ncore_diff; 
ncl = nsi; 
dx = spatial_window/Nx; 
x = (-Nx/2:Nx/2-1)*dx;
[X, Y] = meshgrid(x, x);

epsilon = (max(ncl, nco - (extra_params.ncore_diff)*(sqrt(X.^2+Y.^2)/radius).^extra_params.alpha)).^2;

end