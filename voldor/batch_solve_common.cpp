
#include <stdlib.h>

static int sample_draw(int n)
{
    return (int)(((float)rand() / (float)RAND_MAX) * (n - 1));
}

void sample(int n, int k, int* chosen)
{
    int i = 0;
    while (i < k)
    {
        int s = sample_draw(n);
        if (n >= k)
        {
            int j;
            for (j = 0; j < i; ++j)
            {
                if (s == chosen[j]) { break; }
            }
            if (j < i) { continue; }
        }
        chosen[i] = s;
        i++;
    }
}
