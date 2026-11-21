"""Pixel operations shared by the CUDA and Metal frame kernels."""

MEDIAN = r"""
    float sorted[K*K];
    int n = 0;
    for (int dy=-K/2; dy<=K/2; ++dy) {
        int row = max(0,min(h-1,y+dy));
        for (int dx=-K/2; dx<=K/2; ++dx) {
            float v = src[base + row*w + max(0,min(w-1,x+dx))];
            int i=n++;
            while (i>0 && sorted[i-1]>v) {sorted[i]=sorted[i-1]; --i;}
            sorted[i]=v;
        }
    }
    dst[base+y*w+x]=sorted[K*K/2];
"""

DISK = r"""
    float v = src[base+y*w+x];
    for (int dy=max(-r,-y); dy<=min(r,h-1-y); ++dy) {
        const int span = widths[r+dy];
        const int row = base+(y+dy)*w+x;
        for (int dx=max(-span,-x); dx<=min(span,w-1-x); ++dx)
            v = REDUCE(v,src[row+dx]);
    }
    dst[base+y*w+x]=v;
"""
