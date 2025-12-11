
#include "lock.h"

// OK
Cleaner::Cleaner(std::function<void()> f)
{
    m_f = f;
    m_enable = true;
}

// OK
Cleaner::~Cleaner()
{
    if (m_enable) { m_f(); }
}

// OK
void Cleaner::Set(bool enable)
{
    m_enable = enable;
}
