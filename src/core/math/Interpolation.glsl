float SmoothStep(float a, float b, float x) { return mix(a, b, x * x * (3.0 - 2.0 * x)); }
vec4 SmoothStep(vec4 a, vec4 b, float x)    { return mix(a, b, x * x * (3.0 - 2.0 * x)); }
float SmoothStep(float x)                   { return mix(0.0, 1.0, x * x * (3.0 - 2.0 * x)); }

float PaddedSmoothStep(float x, float a, float b)
{ 
    return SmoothStep(saturate(x * (a + b + 1.0) - a));
}

float PaddedSmoothStep(float x, float a)
{
    return PaddedSmoothStep(x, a, a);
}

float Impulse(float x, float axis, float stdDev)
{
    return exp(-sqr((x - axis) / stdDev));
}

float AnisotropicImpulse(float x, float axis, float stdDevA, float stdDevB)
{
    float impulse = (x < axis) ? exp(-sqr((x - axis) / stdDevA)) : exp(-sqr((x - axis) / stdDevB));
    return saturate((impulse - 0.05) / (1.0 - 0.05));
}

float KickDrop(float t, vec2 p0, vec2 p1, vec2 p2, vec2 p3)
{
    if(t < p1.x)
    {
        return mix(p0.y, p1.y, max(0.0, exp(-sqr((t - p1.x)*2.145966026289347/(p1.x-p0.x))) - 0.01) / 0.99);
    }
    else if(t < p2.x)
    {
        return mix(p1.y, p2.y, (t - p1.x) / (p2.x - p1.x));
    }
    else
    {  
        return mix(p3.y, p2.y, max(0.0, exp(-sqr((t - p2.x)*2.145966026289347/(p3.x-p2.x))) - 0.01) / 0.99);
    }
}

float KickDrop(float t, vec2 p0, vec2 p1, vec2 p2)
{
    return KickDrop(t, p0, p1, p1, p2);
}