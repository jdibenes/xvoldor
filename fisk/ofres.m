
path_base = 'C:/Users/jcds/Documents/GitHub/xvoldor/demo/data/MPI-Sintel/training';
model = 'ptl-pwcnet';

sequences = [
    "alley_1";
    "alley_2";
    "ambush_2";
    "ambush_4";
    "ambush_5";
    "ambush_6";
    "ambush_7";
    "bamboo_1";
    "bamboo_2";
    "bandage_1";
    "bandage_2";
    "cave_2";
    "cave_4";
    "market_2";
    "market_5";
    "market_6";
    "mountain_1";
    "shaman_2";
    "shaman_3";
    "sleeping_1";
    "sleeping_2";
    "temple_2";
    "temple_3";
];

res = [];
mag = [];

for sequence = sequences.'
    disp(sequence)
    path_flo = [path_base '/' 'res_' model '/' char(sequence)];
    files_flo = dir(fullfile(path_flo, '*.flo'));
    for file_flo = files_flo.'
        filename_flo = [path_flo '/' file_flo.name];
        data_flo = load_flo(filename_flo);
        mag_of_err = data_flo(:, :, 1);
        mag_of_flo = data_flo(:, :, 2);
        res = [res; mag_of_err(:)];
        mag = [mag; mag_of_flo(:)];
    end
end

save(['residuals_' model '.mat'], 'res', 'mag', '-v7.3');
