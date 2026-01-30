
#pragma once

// OK
template <typename T>
T clamp(T v, T l, T r)
{
    return (v <= l) ? l : ((v >= r) ? r : v);
}
