function plotcube(x,y,cube,xrange)
% plotcube(x,y,cube,xrange)
%
%   x(ny+1,ny+1,6)                     :: x of cell corners for each face
%   y(ny+1,ny+1,6)                     :: y of cell corners for each face
%   cube(ny,6,ny) or cube(ny*6,ny)     :: pseudocolor data
%   cube(ny,6,ny,3) or cube(ny*6,ny,3) :: RGB data
%   xrange                             :: range of (or max.) x to plot (default [0 420])

  if nargin<4
    xrange=[0 420];
  elseif length(xrange)<2
    xrange=[0 xrange];
  end
  maxlon=xrange(2);
  yrange=[-90 90];

  if size(cube,1)~=size(cube,3)
    % reshape 3060x510xnc -> 510x6x510xnc
    cube = reshape(cube, [size(cube,2),6,size(cube,2),size(cube,3)]);
  end
  [nx nf ny nc]=size(cube);
  x = mod(x, 360);
  z = zeros(size(x))-.01;
  view(2);
  axis([xrange yrange]);
  fc=1; ix=nx/2+1:nx+1;iy=1:ny+1; surf(x(ix,iy,fc),                  y(ix,iy,fc), z(ix,iy,fc), zpad(sq(cube(ix(1:end-1),fc,iy(1:end-1),:))),'LineStyle','None');
  view(2);
  axis([xrange yrange]);
  hold on
  fc=1; ix=   1:nx/2+1;iy=1:ny+1; surf(mod(x(ix,iy,fc)-180,360)+180, y(ix,iy,fc), z(ix,iy,fc), zpad(sq(cube(ix(1:end-1),fc,iy(1:end-1),:))),'LineStyle','None');
  fc=2;                           surf(x(:,:,fc),                    y(:,:,fc), z(:,:,fc), zpad(sq(cube(:,fc,:,:))),'LineStyle','None');
  % arctic [0 180]
  fc=3; ix=1:nx+1;iy=   1:ny/2+1; surf(x(ix,iy,fc),                  y(ix,iy,fc), z(ix,iy,fc), zpad(sq(cube(ix(1:end-1),fc,iy(1:end-1),:))),'LineStyle','None');
  % arctic [-180 0]
  fc=3; ix=1:nx+1;iy=ny/2+1:ny+1; surf(mod(x(ix,iy,fc)-180,360)+180, y(ix,iy,fc), z(ix,iy,fc), zpad(sq(cube(ix(1:end-1),fc,iy(1:end-1),:))),'LineStyle','None');

  % fix cells toughing the pole
  fc=3;
  xn=[x(nx/2,ny/2+1,fc) x(nx/2+1,ny/2,fc) x(nx/2+2,ny/2+1,fc) x(nx/2+1,ny/2+2,fc) x(nx/2,ny/2+1,fc)+360 x(nx/2+1,ny/2,fc)+360 x(nx/2+2,ny/2+1,fc)+360];
  yn=[y(nx/2,ny/2+1,fc) y(nx/2+1,ny/2,fc) y(nx/2+2,ny/2+1,fc) y(nx/2+1,ny/2+2,fc) y(nx/2,ny/2+1,fc) y(nx/2+1,ny/2,fc) y(nx/2+2,ny/2+1,fc)];
  zn=[z(nx/2,ny/2+1,fc) z(nx/2+1,ny/2,fc) z(nx/2+2,ny/2+1,fc) z(nx/2+1,ny/2+2,fc) z(nx/2,ny/2+1,fc) z(nx/2+1,ny/2,fc) z(nx/2+2,ny/2+1,fc)];
  cn=[cube(nx/2,fc,ny/2) cube(nx/2+1,fc,ny/2) cube(nx/2+1,fc,ny/2+1) cube(nx/2,fc,ny/2+1) cube(nx/2,fc,ny/2) cube(nx/2+1,fc,ny/2)];
  xn=[xn' xn'];
  yn=[yn' yn'.*0+90];
  zn=[zn' zn'];
  cn=cn';
  surf(xn, yn, zn, zpad(cn), 'LineStyle','None');

  % pacific
  fc=4;                           surf(x(:,:,fc),                    y(:,:,fc), z(:,:,fc), zpad(sq(cube(:,fc,:,:))), 'LineStyle','None');
  fc=5;                           surf(x(:,:,fc),                    y(:,:,fc), z(:,:,fc), zpad(sq(cube(:,fc,:,:))),'LineStyle','None');
  fc=6; ix=   1:nx/2+1;iy=1:ny+1; surf(mod(x(ix,iy,fc)-180,360)+180, y(ix,iy,fc), z(ix,iy,fc), zpad(sq(cube(ix(1:end-1),fc,iy(1:end-1),:))),'LineStyle','None');
  fc=6; ix=nx/2+1:nx+1;iy=1:ny+1; surf(mod(x(ix,iy,fc)+179,360)-179, y(ix,iy,fc), z(ix,iy,fc), zpad(sq(cube(ix(1:end-1),fc,iy(1:end-1),:))),'LineStyle','None');

  if maxlon>360
  fc=1; ix=nx/2+1:nx+1;iy=1:ny+1; surf(x(ix,iy,fc)+360             , y(ix,iy,fc), z(ix,iy,fc), zpad(sq(cube(ix(1:end-1),fc,iy(1:end-1),:))),'LineStyle','None');
  fc=3; ix=1:nx/2+1;iy= 1:ny/2+1; surf(x(ix,iy,fc)+360             , y(ix,iy,fc), z(ix,iy,fc), zpad(sq(cube(ix(1:end-1),fc,iy(1:end-1),:))),'LineStyle','None');
% fc=3; ix=1:nx+1;iy=ny/2+1:ny+1; surf(mod(x(ix,iy,fc)-180,360)+540, y(ix,iy,fc), z(ix,iy,fc), zpad(sq(cube(ix(1:end-1),fc,iy(1:end-1),:))),'LineStyle','None');
% fc=6; ix=   1:nx/2+1;iy=1:ny+1; surf(mod(x(ix,iy,fc)-180,360)+540, y(ix,iy,fc), z(ix,iy,fc), zpad(sq(cube(ix(1:end-1),fc,iy(1:end-1),:))),'LineStyle','None');
  fc=6; ix=nx/2+1:nx+1;iy=ny/2+1:ny+1; surf(mod(x(ix,iy,fc)+179,360)+181, y(ix,iy,fc), z(ix,iy,fc), zpad(sq(cube(ix(1:end-1),fc,iy(1:end-1),:))),'LineStyle','None');
  end
  if maxlon>405
  fc=2;                           surf(x(:,:,fc)+360,                y(:,:,fc), z(:,:,fc), zpad(sq(cube(:,fc,:,:))),'LineStyle','None');
  end

  % fix cells toughing the pole
  fc=6;
  xn=[x(nx/2+1,ny/2+2,fc) x(nx/2+2,ny/2+1,fc) x(nx/2+1,ny/2,fc) x(nx/2,ny/2+1,fc) x(nx/2+1,ny/2+2,fc)+360 x(nx/2+2,ny/2+1,fc)+360];
  yn=[y(nx/2+1,ny/2+2,fc) y(nx/2+2,ny/2+1,fc) y(nx/2+1,ny/2,fc) y(nx/2,ny/2+1,fc) y(nx/2+1,ny/2+2,fc) y(nx/2+2,ny/2+1,fc)];
  zn=[z(nx/2+1,ny/2+2,fc) z(nx/2+2,ny/2+1,fc) z(nx/2+1,ny/2,fc) z(nx/2,ny/2+1,fc) z(nx/2+1,ny/2+2,fc) z(nx/2+2,ny/2+1,fc)];
  cn=[cube(nx/2+1,fc,ny/2+1) cube(nx/2+1,fc,ny/2) cube(nx/2,fc,ny/2) cube(nx/2,fc,ny/2+1) cube(nx/2+1,fc,ny/2+1)];
  xn=[xn' xn'];
  yn=[yn' yn'.*0-90];
  zn=[zn' zn'];
  cn=cn';
  surf(xn, yn, zn, zpad(cn), 'LineStyle','None');

  hold off
  view(2);
  axis([xrange yrange]);
end

