
#pragma once

#include <functional>

class Cleaner
{
private:
    std::function<void()> m_f;
    bool m_enable;

public:
    Cleaner(std::function<void()> f);
    ~Cleaner();
    void Set(bool enable);
};
