# Kernel benchmarks

GPU: Tesla T4

| kernel | shape | impl | ms | gbps | pct_peak | speedup |
|---|---|---|---|---|---|---|
| layernorm | 128x768 | pytorch | 0.018 | 43.977 | 13.743 | 1.000 |
| layernorm | 128x768 | v1 block+smem | 0.015 | 54.507 | 17.033 | 1.239 |
| layernorm | 128x768 | v2 warp+float4 | 0.010 | 76.619 | 23.944 | 1.742 |
| layernorm | 1024x768 | pytorch | 0.052 | 121.050 | 37.828 | 1.000 |
| layernorm | 1024x768 | v1 block+smem | 0.076 | 82.835 | 25.886 | 0.684 |
| layernorm | 1024x768 | v2 warp+float4 | 0.032 | 198.487 | 62.027 | 1.640 |
| layernorm | 4096x768 | pytorch | 0.171 | 147.512 | 46.098 | 1.000 |
| layernorm | 4096x768 | v1 block+smem | 0.274 | 91.929 | 28.728 | 0.623 |
| layernorm | 4096x768 | v2 warp+float4 | 0.116 | 216.530 | 67.666 | 1.468 |
| layernorm | 16384x768 | pytorch | 0.494 | 203.825 | 63.695 | 1.000 |
| layernorm | 16384x768 | v1 block+smem | 0.624 | 161.303 | 50.407 | 0.791 |
| layernorm | 16384x768 | v2 warp+float4 | 0.439 | 229.484 | 71.714 | 1.126 |
| bias_gelu | 128x3072 | pytorch (add + gelu) | 0.022 | 144.268 | 45.084 | 1.000 |
| bias_gelu | 128x3072 | v1 scalar | 0.010 | 330.613 | 103.317 | 2.292 |
| bias_gelu | 128x3072 | v2 float4 | 0.009 | 353.644 | 110.514 | 2.451 |
| bias_gelu | 1024x3072 | pytorch (add + gelu) | 0.218 | 115.589 | 36.121 | 1.000 |
| bias_gelu | 1024x3072 | v1 scalar | 0.104 | 242.008 | 75.627 | 2.094 |
| bias_gelu | 1024x3072 | v2 float4 | 0.108 | 234.085 | 73.152 | 2.025 |
| bias_gelu | 4096x3072 | pytorch (add + gelu) | 0.854 | 117.840 | 36.825 | 1.000 |
| bias_gelu | 4096x3072 | v1 scalar | 0.404 | 249.350 | 77.922 | 2.116 |
| bias_gelu | 4096x3072 | v2 float4 | 0.418 | 240.857 | 75.268 | 2.044 |
| bias_gelu | 16384x3072 | pytorch (add + gelu) | 3.399 | 118.460 | 37.019 | 1.000 |
| bias_gelu | 16384x3072 | v1 scalar | 1.606 | 250.795 | 78.373 | 2.117 |
| bias_gelu | 16384x3072 | v2 float4 | 1.680 | 239.715 | 74.911 | 2.024 |
| softmax | 1x12x128x128 | pytorch (scale + mask + softmax) | 0.061 | 25.735 | 8.042 | 1.000 |
| softmax | 1x12x128x128 | v1 block+smem | 0.029 | 53.891 | 16.841 | 2.094 |
| softmax | 1x12x128x128 | v2 warp online | 0.008 | 186.592 | 58.310 | 7.250 |
| softmax | 8x12x128x128 | pytorch (scale + mask + softmax) | 0.226 | 55.752 | 17.422 | 1.000 |
| softmax | 8x12x128x128 | v1 block+smem | 0.243 | 51.768 | 16.177 | 0.929 |
| softmax | 8x12x128x128 | v2 warp online | 0.049 | 255.577 | 79.868 | 4.584 |
| softmax | 4x12x512x512 | pytorch (scale + mask + softmax) | 1.744 | 57.732 | 18.041 | 1.000 |
| softmax | 4x12x512x512 | v1 block+smem | 0.637 | 158.086 | 49.402 | 2.738 |
| softmax | 4x12x512x512 | v2 warp online | 0.384 | 262.078 | 81.899 | 4.540 |
| softmax | 8x12x1024x1024 | pytorch (scale + mask + softmax) | 14.392 | 55.956 | 17.486 | 1.000 |
| softmax | 8x12x1024x1024 | v1 block+smem | 3.448 | 233.545 | 72.983 | 4.174 |
| softmax | 8x12x1024x1024 | v2 warp online | 3.371 | 238.923 | 74.664 | 4.270 |
| attention | 1x12x128x64 | pytorch unfused | 0.207 | 7.584 | 2.370 | 1.000 |
| attention | 1x12x128x64 | pytorch sdpa | 0.031 | 50.017 | 15.630 | 6.595 |
| attention | 1x12x128x64 | matmul + v2 softmax | 0.085 | 18.500 | 5.781 | 2.439 |
| attention | 1x12x128x64 | flash (fused) | 0.062 | 25.233 | 7.885 | 3.327 |
| attention | 8x12x128x64 | pytorch unfused | 0.392 | 32.087 | 10.027 | 1.000 |
| attention | 8x12x128x64 | pytorch sdpa | 0.156 | 80.462 | 25.144 | 2.508 |
| attention | 8x12x128x64 | matmul + v2 softmax | 0.225 | 55.887 | 17.465 | 1.742 |
| attention | 8x12x128x64 | flash (fused) | 0.396 | 31.780 | 9.931 | 0.990 |
| attention | 4x12x512x64 | pytorch unfused | 2.701 | 9.316 | 2.911 | 1.000 |
| attention | 4x12x512x64 | pytorch sdpa | 0.743 | 33.866 | 10.583 | 3.635 |
| attention | 4x12x512x64 | matmul + v2 softmax | 1.525 | 16.498 | 5.156 | 1.771 |
| attention | 4x12x512x64 | flash (fused) | 2.068 | 12.166 | 3.802 | 1.306 |
| attention | 8x12x1024x64 | pytorch unfused | 21.370 | 4.711 | 1.472 | 1.000 |
| attention | 8x12x1024x64 | pytorch sdpa | 4.896 | 20.560 | 6.425 | 4.365 |
| attention | 8x12x1024x64 | matmul + v2 softmax | 12.340 | 8.157 | 2.549 | 1.732 |
| attention | 8x12x1024x64 | flash (fused) | 15.318 | 6.572 | 2.054 | 1.395 |
| attention_decode | 1x12x1x512 | pytorch unfused | 0.190 | 16.598 | 5.187 | 1.000 |
| attention_decode | 1x12x1x512 | pytorch sdpa | 0.085 | 37.176 | 11.618 | 2.240 |
| attention_decode | 1x12x1x512 | flash (fused) | 0.049 | 64.050 | 20.016 | 3.859 |
| attention_decode | 8x12x1x512 | pytorch unfused | 0.186 | 135.297 | 42.280 | 1.000 |
| attention_decode | 8x12x1x512 | pytorch sdpa | 0.359 | 70.241 | 21.950 | 0.519 |
| attention_decode | 8x12x1x512 | flash (fused) | 0.101 | 249.188 | 77.871 | 1.842 |
| attention_decode | 8x12x1x1024 | pytorch unfused | 0.247 | 204.349 | 63.859 | 1.000 |
| attention_decode | 8x12x1x1024 | pytorch sdpa | 0.697 | 72.234 | 22.573 | 0.353 |
| attention_decode | 8x12x1x1024 | flash (fused) | 0.200 | 252.143 | 78.795 | 1.234 |
