function[bi] =create_random_birefringence_InputModuleL1(type,z,L,Lc,Lspun,bi_old)


if strcmp(type,'No random')
    bi = 0*z;
end;

if strcmp(type,'Given from extern')
    bi=bi_old;
end
    
if strcmp(type,'Random')
    angles=[0 2*pi*(rand(1,floor(L/Lc))-0.5)];
    bi=interp1(linspace(0,max(z),length(angles)),angles,z,'cubic');
    bi = bi + 2*pi*z/Lspun;
end
    
 
figure(1),plot(z,bi,'b');
pause(1)