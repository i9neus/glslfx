// *******************************************************************************************************
//    Half-precision float
// *******************************************************************************************************

#define half uint
//#define USE_REFERENCE_FLOAT_PACKING

#ifdef USE_REFERENCE_FLOAT_PACKING

    half FloatToHalf(float f)
    {
        float exponent = ceil(log(abs(f)));
        int mantissa = int(float((1 << 10) - 1) * (abs(f) / exp(exponent)));

        return (half((sign(f) + 1.0) * 0.5) << 15u) | half((int(exponent) + 15) << 10) | half(mantissa);
    }

    float HalfToFloat(half h)
    {
        float exponent = float(((int(h) >> 10) & ((1 << 5) - 1)) - 15);
        float mantissa = float(int(h) & ((1 << 10) - 1)) / float((1 << 10) - 1);
        float sgn = float((int(h) >> 15) & 1) * 2.0 - 1.0;

        return mantissa * exp(exponent) * sgn;
    }

    float PackFloats(in float a, in float b)
    {
        return uintBitsToFloat((FloatToHalf(a) << 16) | FloatToHalf(b));
    }

    void UnpackFloats(in float packed, out float a, out float b)
    {
        a = HalfToFloat(floatBitsToUint(packed) >> 16);
        b = HalfToFloat(floatBitsToUint(packed) & uint((1 << 16) - 1));
    }
    
#else

    float PackFloats(in float a, in float b)
    {
        return uintBitsToFloat(packHalf2x16(vec2(a, b)));
    }

    float PackFloats(in vec2 v)
    {
        return uintBitsToFloat(packHalf2x16(v));
    }

    void UnpackFloats(in float i, out float a, out float b)
    {
        vec2 v = unpackHalf2x16(floatBitsToUint(i));
        a = v.x; b = v.y;
    }

    vec2 UnpackFloats(in float i)
    {
        return unpackHalf2x16(floatBitsToUint(i));
    }
    
#endif