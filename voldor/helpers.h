
#pragma once

// OK
template <typename T>
T clamp(T v, T l, T r)
{
    return (v <= l) ? l : ((v >= r) ? r : v);
}

// OK
template <typename T>
T acos_small(T x)
{
    return 2 * std::atan2(std::sqrt(1 - x), std::sqrt(1 + x));
}
