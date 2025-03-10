
function data = load_flo(filename)
fid = fopen(filename, 'rb');
fread(fid, 1, 'single');
width = fread(fid, 1, 'uint32');
height = fread(fid, 1, "uint32");
data = fread(fid, [2, width*height], 'single');
fclose(fid);
data = reshape(data, [2, width, height]);
data = permute(data, [3, 2, 1]);
end
