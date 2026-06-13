
#pragma once

int load_file(char const* filename, void* buffer, int offset, int count);
void load_flow(char const* filename, float* buffer, int buffer_index, int* width, int* height, bool strict, int check_width, int check_height);
void load_pose_hl2ss(char const* filename, float* buffer);
void load_window_flow(char const* path, char const* pattern, int start, int stop, float* buffer, bool strict, int width, int height);
void load_window_pose_hl2ss(char const* path, char const* pattern, int start, int stop, float* buffer);
