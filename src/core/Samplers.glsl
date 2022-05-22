const mat4 kOrderedDither = mat4(vec4(0.0, 8.0, 2.0, 10.), vec4(12., 4., 14., 6.), vec4(3., 11., 1., 9.), vec4(15., 7., 13., 5.));
float OrderedDither()
{    
    return (kOrderedDither[int(gFragCoord.x) & 3][int(gFragCoord.y) & 3] + 1.0) / 17.0;
}

vec2 SampleCapitulum(ivec2 sampleIdx)
{
    #define kGoldenAngle 2.399963229728653
    vec2 xi = Rand().xy;
    float phi = kGoldenAngle * (float(sampleIdx.x) + xi.x);
    float radius = sqrt(mix(float(sampleIdx.x), float(sampleIdx.x+3), xi.y) / float(sampleIdx.y - 1));
    
    return vec2(cos(phi), sin(phi)) * radius;
}

vec2 SampleUnitDisc(vec2 xi)
{
    float phi = xi.y * kTwoPi;   
    return vec2(sin(phi), cos(phi)) * sqrt(xi.x);   
}

vec3 SampleUnitSphere(vec2 xi)
{
    xi.x = xi.x * 2.0 - 1.0;
    xi.y *= kTwoPi;

    float sinTheta = sqrt(1.0 - xi.x * xi.x);
    return vec3(cos(xi.y) * sinTheta, xi.x, sin(xi.y) * sinTheta);
}