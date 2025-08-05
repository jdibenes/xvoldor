

#include <stdio.h>
#include <string>
#include <stdexcept>


void load_file(char const* filename, void* buffer, int offset, int count)
{
    FILE* f = fopen(filename, "rb");
    if (!f) { throw std::runtime_error(std::string("Failed to open file: ") + filename); }
    int bytes;
    if (count < 0)
    {
        fseek(f, 0, SEEK_END);
        bytes = ftell(f) - offset;
    }
    else
    {
        bytes = count;
    }
    fseek(f, offset, SEEK_SET);
    int bytes_read = fread(buffer, 1, bytes, f);
    fclose(f);
    if (bytes_read != bytes) { throw std::runtime_error("Read " + std::to_string(bytes_read) + "/" + std::to_string(bytes) + " from file: " + filename); }
}
