

name = "/home/kim/Desktop/ssd/research3/data_vimeo/38";

im1 = double(imread(strcat(name, "/im1.png")));
im2 = double(imread(strcat(name, "/im2.png")));
im3 = double(imread(strcat(name, "/im3.png")));


name = "/home/kim/ssd/snufilm-test/YouTube_test/0003";

im1 = double(imread(strcat(name, "/00293.png")));
im2 = double(imread(strcat(name, "/00301.png")));
im3 = double(imread(strcat(name, "/00309.png")));


imshow(im1/255);
% imshow(im3/255);