
#pragma once

#include <vector>

template <typename T, int n> struct add_vector {
public:
    typedef typename add_vector<std::vector<T>, n - 1>::type type;
};

template <typename T> struct add_vector<T, 0>
{
public:
    typedef T type;
};

template <typename T> struct remove_vector {
public:
    typedef T type;
};

template <typename T> struct remove_vector<std::vector<T>> {
    typedef typename remove_vector<T>::type type;
};
