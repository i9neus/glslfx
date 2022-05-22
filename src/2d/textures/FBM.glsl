// Fractional Brownian motion
float FBM(in vec3 p, in int octaves)
{
    float F = 0.0;
    float sumWeights = 0.0;
    for(int i = 0; i < octaves; ++i)
    {
        float exponent = pow(2.0, float(i));
        vec3 fp = fract(p * exponent);
        uvec3 ip = uvec3(p * exponent);     
        
        float x0 = SmoothStep(UintToFloat(HashOf(ip[0], ip[1], ip[2])), UintToFloat(HashOf(ip[0] + 1u, ip[1], ip[2])), fp[0]);
        float x1 = SmoothStep(UintToFloat(HashOf(ip[0], ip[1] + 1u, ip[2])), UintToFloat(HashOf(ip[0] + 1u, ip[1] + 1u, ip[2])), fp[0]);
        float x2 = SmoothStep(UintToFloat(HashOf(ip[0], ip[1], ip[2] + 1u)), UintToFloat(HashOf(ip[0] + 1u, ip[1], ip[2] + 1u)), fp[0]);
        float x3 = SmoothStep(UintToFloat(HashOf(ip[0], ip[1] + 1u, ip[2] + 1u)), UintToFloat(HashOf(ip[0] + 1u, ip[1] + 1u, ip[2] + 1u)), fp[0]);
        
        F += SmoothStep(SmoothStep(x0, x1, fp[1]), SmoothStep(x2, x3, fp[1]), fp[2]) / exponent;
        sumWeights += 1.0 / exponent;        
    }
    return F / sumWeights;
}