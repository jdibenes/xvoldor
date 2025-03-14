
path_base = 'C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/kitti-flow/training/res_occ';
model = 'searaft';

res = [];
mag = [];

path_flo = [path_base '/' 'res_' model];
files_flo = dir(fullfile(path_flo, '*.flo'));
for file_flo = files_flo.'
    filename_flo = [path_flo '/' file_flo.name];
    data_flo = load_flo(filename_flo);
    mag_of_err = data_flo(:, :, 1);
    mag_of_flo = data_flo(:, :, 2);
    res = [res; mag_of_err(:)];
    mag = [mag; mag_of_flo(:)];
end

select = mag > 0;
res = res(select);
mag = mag(select);

save(['residuals_kitti_' model '.mat'], 'res', 'mag', '-v7.3');
