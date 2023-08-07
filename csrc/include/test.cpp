#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <cstdint>
#include<sched.h>

inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32 = {w};
    return fp32.as_value;
}

inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32 = {f};
    return fp32.as_bits;
}

// fp32 -> fp16
inline uint16_t fp16_ieee_from_fp32_value(float f) {
    // const float scale_to_inf = 0x1.0p+112f;
    // const float scale_to_zero = 0x1.0p-110f;
    uint32_t scale_to_inf_bits = (uint32_t) 239 << 23;
    uint32_t scale_to_zero_bits = (uint32_t) 17 << 23;
    float scale_to_inf_val, scale_to_zero_val;
    std::memcpy(&scale_to_inf_val, &scale_to_inf_bits, sizeof(scale_to_inf_val));
    std::memcpy(&scale_to_zero_val, &scale_to_zero_bits, sizeof(scale_to_zero_val));
    const float scale_to_inf = scale_to_inf_val;
    const float scale_to_zero = scale_to_zero_val;

    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

          const uint32_t w = (uint32_t)fp32_to_bits(f);
          const uint32_t shl1_w = w + w;
          const uint32_t sign = w & UINT32_C(0x80000000);
          uint32_t bias = shl1_w & UINT32_C(0xFF000000);
          if (bias < UINT32_C(0x71000000)) {
                  bias = UINT32_C(0x71000000);
          }

          base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
          const uint32_t bits = (uint32_t)fp32_to_bits(base);
          const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
          const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
          const uint32_t nonsign = exp_bits + mantissa_bits;
          return static_cast<uint16_t>(
            (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign)
          );
}

int main() {
    float f = 2.71828;
    uint16_t bf16_val = fp16_ieee_from_fp32_value(f);

    std::cout << "Input: " << std::setprecision(7) << f << std::endl;
    std::cout << "Output: " << bf16_val << std::endl;

    return 0;
}


// // fp32 -> fp16
// inline uint16_t fp16_ieee_from_fp32_value(float f) {
//     // const float scale_to_inf = 0x1.0p+112f;
//     // const float scale_to_zero = 0x1.0p-110f;
//     uint32_t scale_to_inf_bits = (uint32_t) 239 << 23;
//     uint32_t scale_to_zero_bits = (uint32_t) 17 << 23;
//     float scale_to_inf_val, scale_to_zero_val;
//     std::memcpy(&scale_to_inf_val, &scale_to_inf_bits, sizeof(scale_to_inf_val));
//     std::memcpy(&scale_to_zero_val, &scale_to_zero_bits, sizeof(scale_to_zero_val));
//     const float scale_to_inf = scale_to_inf_val;
//     const float scale_to_zero = scale_to_zero_val;

//     float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

//           const uint32_t w = (uint32_t)fp32_to_bits(f);
//           const uint32_t shl1_w = w + w;
//           const uint32_t sign = w & UINT32_C(0x80000000);
//           uint32_t bias = shl1_w & UINT32_C(0xFF000000);
//           if (bias < UINT32_C(0x71000000)) {
//                   bias = UINT32_C(0x71000000);
//           }

//           base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
//           const uint32_t bits = (uint32_t)fp32_to_bits(base);
//           const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
//           const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
//           const uint32_t nonsign = exp_bits + mantissa_bits;
//           return static_cast<uint16_t>(
//             (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign)
//           );
//   }