#define kMLPNumLayers          2
#define kMLPWidth              4
#define kMLPOutputLayer        kMLPNumLayers

#define kMLPWeightStride       (kMLPWidth*kMLPWidth/4)
#define kMLPBiasStride         (kMLPWidth/4)
#define kMLPActStride          (kMLPWidth/4)
#define kMLPErrorStride        (kMLPWidth/4)
#define kMLPLayerStride        (kMLPWeightStride + kMLPBiasStride + kMLPActStride + kMLPErrorStride)
#define kMLPNetStride          (kMLPLayerStride * kMLPNumLayers + 1)
#define kMLPRowStride          (kMLPWidth / 2)
#define kMLPColStride          (kMLPWidth / 2)
#define kMLPQuadsPerCol        (kMLPWidth / 4)

#define kMLPFrameSkip          1

// Relative to the start of the network
#define kMLPLossOffset         (kMLPLayerStride * kMLPNumLayers)

// Relative to the start of the layer
#define kMLPBiasOffset         kMLPWeightStride
#define kMLPActOffset          (kMLPBiasOffset + kMLPBiasStride)
#define kMLPErrorOffset        (kMLPActOffset + kMLPActStride)

#define kFramesPerEpoch        (2 * kMLPNumLayers + 2)
#define kForwardFrame          0
#define kLossFrame             kMLPNumLayers
#define kBackwardFrame         (kLossFrame + 1)

#define kIgnore                -1
#define kNotUpdated            0
#define kUpdated               1

#define GetActIdx(paramIdx)  (paramIdx - kMLPActOffset)
#define GetBiasIdx(paramIdx) (paramIdx - kMLPBiasOffset)
#define GetErrorIdx(paramIdx) (paramIdx - kMLPErrorOffset)
#define IsWeight(paramIdx)   (paramIdx < kMLPBiasOffset)
#define IsBias(paramIdx)     (paramIdx >= kMLPBiasOffset && paramIdx < kMLPActOffset)
#define IsAct(paramIdx)      (paramIdx >= kMLPActOffset && paramIdx < kMLPErrorOffset)
#define IsError(paramIdx)    (paramIdx >= kMLPErrorOffset)
#define PrevLayer(x)         (x - kMLPLayerStride)
#define NextLayer(x)         (x + kMLPLayerStride)

#define kClipView 0
#define kScreenDownsample 1

// *******************************************************************************************************
// Global variables

vec2 gResolution;
vec2 gFragCoord;
uvec4 rngSeed;
float gDxyDuv;

void SetGlobals(vec2 fragCoord, vec2 resolution)
{
    gFragCoord = fragCoord;
    gResolution = resolution;
    
     // First derivative of screen to world space (assuming square pixels)
    gDxyDuv = 1.0 / min(gResolution.x, gResolution.y);
}

bool IsTexelInClipRegion(vec2 xy)
{
#if kClipView == 1
    return (abs(xy.x - (gResolution.x / 2.0)) < gResolution.y / 2.0);
#else
    return true;
#endif
}

// *******************************************************************************************************
//    Math functions
// *******************************************************************************************************

#define kPi                    3.14159265359
#define kInvPi                 (1.0 / 3.14159265359)
#define kTwoPi                 (2.0 * kPi)
#define kHalfPi                (0.5 * kPi)
#define kRoot2                 1.41421356237
#define kFltMax                3.402823466e+38
#define kIntMax                0x7fffffff
#define kOne                   vec3(1.0)
#define kZero                  vec3(0.0)
#define kPink                  vec3(1.0, 0.0, 0.2)

float cubrt(float a)           { return sign(a) * pow(abs(a), 1.0 / 3.0); }
float toRad(float deg)         { return kTwoPi * deg / 360.0; }
float toDeg(float rad)         { return 360.0 * rad / kTwoPi; }
float sqr(float a)             { return a * a; }
vec3 sqr(vec3 a)               { return a * a; }
vec4 sqr(vec4 a)               { return a * a; }
int sqr(int a)                 { return a * a; }
int mod2(int a, int b)         { return ((a % b) + b) % b; }
float mod2(float a, float b)   { return mod(mod(a, b) + b, b); }
vec3 mod2(vec3 a, vec3 b)      { return mod(mod(a, b) + b, b); }
float length2(vec2 v)          { return dot(v, v); }
float length2(vec3 v)          { return dot(v, v); }
int sum(ivec2 a)               { return a.x + a.y; }
float luminance(vec3 v)        { return v.x * 0.17691 + v.y * 0.8124 + v.z * 0.01063; }
float mean(vec3 v)             { return v.x / 3.0 + v.y / 3.0 + v.z / 3.0; }
vec4 mul4(vec3 a, mat4 m)      { return vec4(a, 1.0) * m; }
vec3 mul3(vec3 a, mat4 m)      { return (vec4(a, 1.0) * m).xyz; }
float sin01(float a)           { return 0.5 * sin(a) + 0.5; }
float cos01(float a)           { return 0.5 * cos(a) + 0.5; }
float saturate(float a)        { return clamp(a, 0.0, 1.0); }
vec3 saturate(vec3 a)        { return clamp(a, 0.0, 1.0); }
float saw01(float a)           { return abs(fract(a) * 2.0 - 1.0); }
float cwiseMax(vec3 v)         { return (v.x > v.y) ? ((v.x > v.z) ? v.x : v.z) : ((v.y > v.z) ? v.y : v.z); }
float cwiseMax(vec2 v)         { return (v.x > v.y) ? v.x : v.y; }
float cwiseMin(vec3 v)         { return (v.x < v.y) ? ((v.x < v.z) ? v.x : v.z) : ((v.y < v.z) ? v.y : v.z); }
float cwiseMin(vec2 v)         { return (v.x < v.y) ? v.x : v.y; }
void sort(inout float a, inout float b) { if(a > b) { float s = a; a = b; b = s; } }
void swap(inout float a, inout float b) { float s = a; a = b; b = s; }

vec3 safeAtan(vec3 a, vec3 b)
{
    vec3 r;
    #define kAtanEpsilon 1e-10
    r.x = (abs(a.x) < kAtanEpsilon && abs(b.x) < kAtanEpsilon) ? 0.0 : atan(a.x, b.x); 
    r.y = (abs(a.y) < kAtanEpsilon && abs(b.y) < kAtanEpsilon) ? 0.0 : atan(a.y, b.y); 
    r.z = (abs(a.z) < kAtanEpsilon && abs(b.z) < kAtanEpsilon) ? 0.0 : atan(a.z, b.z); 
    return r;
}

vec4 Sign(vec4 v)
{
    return step(vec4(0.0), v) * 2.0 - 1.0;
}

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

bool QuadraticSolve(float a, float b, float c, out float t0, out float t1)
{
    float b2ac4 = b * b - 4.0 * a * c;
    if(b2ac4 < 0.0) { return false; } 

    float sqrtb2ac4 = sqrt(b2ac4);
    t0 = (-b + sqrtb2ac4) / (2.0 * a);
    t1 = (-b - sqrtb2ac4) / (2.0 * a);    
    return true;
}

// *******************************************************************************************************
//    2D SVG
// *******************************************************************************************************

float SDFLine(vec2 p, vec2 v0, vec2 v1, float thickness)
{
    v1 -= v0;
    float t = saturate((dot(p, v1) - dot(v0, v1)) / dot(v1, v1));
    vec2 perp = v0 + t * v1;
    return saturate((thickness - length(p - perp)) / gDxyDuv);
}

float SDFQuad(vec2 p, vec2 v[4], float thickness)
{
    float c = 0.0;
    for(int i = 0; i < 4; i++)
    {
        c = max(c, SDFLine(p, v[i], v[(i+1)%4], thickness)); 
    }
 
    return c;
}

float SDFCircle(vec2 p, vec2 o, float r, float thickness, bool fill)
{
    float dist = fill ? ((r - length(o - p)) / thickness) : (1.0 - abs(r - length(o - p)) / thickness);
    return saturate(dist);
}

// *******************************************************************************************************
//    2D primitive tests
// *******************************************************************************************************

bool IsPointInQuad(vec2 uv, vec2 v[4])
{
    for(int i = 0; i < 4; i++)
    {
        if(dot(uv - v[i], v[i] - v[(i+1)%4]) > 0.0) { return false; }
    }
    return true;
}

// *******************************************************************************************************
//    Transforms 
// *******************************************************************************************************

mat3 WorldToViewMatrix(float rot, vec2 trans, float sca)
{   
    return mat3(vec3(cos(rot) / sca, sin(rot) / sca, trans.x), 
                vec3(-sin(rot) / sca, cos(rot) / sca, trans.y),
                vec3(0.0, 0.0, 1.0));
}

mat3 WorldToViewMatrix(vec2 trans, float sca)
{   
    return mat3(vec3(1.0 / sca, 0.0, trans.x), 
                vec3(0.0, 1.0 / sca, trans.y),
                vec3(0.0, 0.0, 1.0));
}

vec2 TransformScreenToWorld(vec2 p)
{   
    return (p - vec2(gResolution.xy) * 0.5) / float(gResolution.y); 
}

vec2 TransformScreenToWorld(vec2 p, vec2 o)
{   
    return (p - o * 0.5) / float(gResolution.y); 
}


vec3 Cartesian2DToBarycentric(vec2 p)
{    
    return vec3(p, 0.0) * mat3(vec3(0.0, 1.0 / 0.8660254037844387, 0.0),
                          vec3(1.0, 0.5773502691896257, 0.0),
                          vec3(-1.0, 0.5773502691896257, 0.0));
    
}

// Maps an input uv position to periodic hexagonal tiling
//     inout vec2 uv: The mapped uv coordinate
//     out vec3 bary: The Barycentric coordinates at the point on the hexagon
//     out ivec2 ij: The coordinate of the tile
vec2 Cartesian2DToHexagonalTiling(in vec2 uv, out vec3 bary, out ivec2 ij)
{    
    #define kHexRatio vec2(1.5, 0.8660254037844387)
    vec2 uvClip = mod(uv + kHexRatio, 2.0 * kHexRatio) - kHexRatio;
    
    ij = ivec2((uv + kHexRatio) / (2.0 * kHexRatio)) * 2;
    if(uv.x + kHexRatio.x <= 0.0) ij.x -= 2;
    if(uv.y + kHexRatio.y <= 0.0) ij.y -= 2;
    
    bary = Cartesian2DToBarycentric(uvClip);
    if(bary.x > 0.0)
    {
        if(bary.z > 1.0) { bary += vec3(-1.0, 1.0, -2.0); ij += ivec2(-1, 1); }
        else if(bary.y > 1.0) { bary += vec3(-1.0, -2.0, 1.0); ij += ivec2(1, 1); }
    }
    else
    {
        if(bary.y < -1.0) { bary += vec3(1.0, 2.0, -1.0); ij += ivec2(-1, -1); }
        else if(bary.z < -1.0) { bary += vec3(1.0, -1.0, 2.0); ij += ivec2(1, -1); }
    }

    return vec2(bary.y * 0.5773502691896257 - bary.z * 0.5773502691896257, bary.x);
}

bool InverseSternograph(inout vec2 uv, float zoom)
{
    float theta = length(uv) * kPi * zoom;
    if(theta >= kPi - 1e-1) { return false; }
    
    float phi = atan(-uv.y, -uv.x) + kPi;
    
    vec3 sph = vec3(cos(phi) * sin(theta), sin(phi) * sin(theta), -cos(theta));
    
    uv = vec2(sph.x / (1.0 - sph.z), sph.y / (1.0 - sph.z));
    return true;
}

// *******************************************************************************************************
//    Random number generation
// *******************************************************************************************************

// Permuted congruential generator from "Hash Functions for GPU Rendering" (Jarzynski and Olano)
// http://jcgt.org/published/0009/03/02/paper.pdf
uvec4 PCGAdvance()
{
    rngSeed = rngSeed * 1664525u + 1013904223u;
    
    rngSeed.x += rngSeed.y*rngSeed.w; 
    rngSeed.y += rngSeed.z*rngSeed.x; 
    rngSeed.z += rngSeed.x*rngSeed.y; 
    rngSeed.w += rngSeed.y*rngSeed.z;
    
    rngSeed ^= rngSeed >> 16u;
    
    rngSeed.x += rngSeed.y*rngSeed.w; 
    rngSeed.y += rngSeed.z*rngSeed.x; 
    rngSeed.z += rngSeed.x*rngSeed.y; 
    rngSeed.w += rngSeed.y*rngSeed.z;
    
    return rngSeed;
}

// Generates a tuple of canonical random number and uses them to sample an input texture
vec4 Rand(sampler2D sampler)
{
    return texelFetch(sampler, (ivec2(gFragCoord) + ivec2(PCGAdvance() >> 16)) % 1024, 0);
}

// Generates a tuple of canonical random numbers in the range [0, 1]
vec4 Rand()
{
    return vec4(PCGAdvance()) / float(0xffffffffu);
}

// Seed the PCG hash function with the current frame multipled by a prime
void PCGInitialise(uint frame)
{    
    rngSeed = uvec4(20219u, 7243u, 12547u, 28573u) * frame;
}

// Reverse the bits of 32-bit inteter
uint RadicalInverse(uint i)
{
    i = ((i & 0xffffu) << 16u) | (i >> 16u);
    i = ((i & 0x00ff00ffu) << 8u) | ((i & 0xff00ff00u) >> 8u);
    i = ((i & 0x0f0f0f0fu) << 4u) | ((i & 0xf0f0f0f0u) >> 4u);
    i = ((i & 0x33333333u) << 2u) | ((i & 0xccccccccu) >> 2u);    
    i = ((i & 0x55555555u) << 1u) | ((i & 0xaaaaaaaau) >> 1u);        
    return i;
}

// Samples the radix-2 Halton sequence from seed value, i
float HaltonBase2(uint i)
{    
    return float(RadicalInverse(i)) / float(0xffffffffu);
}

// *******************************************************************************************************
//    Hash functions
// *******************************************************************************************************

// Constants for the Fowler-Noll-Vo hash function
// https://en.wikipedia.org/wiki/Fowler-Noll-Vo_hash_function
#define kFNVPrime              0x01000193u
#define kFNVOffset             0x811c9dc5u
#define kDimsPerBounce         4

// Mix and combine two hashes
uint HashCombine(uint a, uint b)
{
    return (((a << (31u - (b & 31u))) | (a >> (b & 31u)))) ^
            ((b << (a & 31u)) | (b >> (31u - (a & 31u))));
}

// Compute a 32-bit Fowler-Noll-Vo hash for the given input
uint HashOf(uint i)
{
    uint h = (kFNVOffset ^ (i & 0xffu)) * kFNVPrime;
    h = (h ^ ((i >> 8u) & 0xffu)) * kFNVPrime;
    h = (h ^ ((i >> 16u) & 0xffu)) * kFNVPrime;
    h = (h ^ ((i >> 24u) & 0xffu)) * kFNVPrime;
    return h;
}

uint HashOf(int a) { return HashOf(uint(a)); }
uint HashOf(uint a, uint b) { return HashCombine(HashOf(a), HashOf(b)); }
uint HashOf(uint a, uint b, uint c) { return HashCombine(HashCombine(HashOf(a), HashOf(b)), HashOf(c)); }
uint HashOf(uint a, uint b, uint c, uint d) { return HashCombine(HashCombine(HashOf(a), HashOf(b)), HashCombine(HashOf(c), HashOf(d))); }
uint HashOf(vec2 v) { return HashCombine(HashOf(uint(v.x)), HashOf(uint(v.y))); }
uint HashOf(ivec2 v) { return HashCombine(HashOf(uint(v.x)), HashOf(uint(v.y))); }

// Samples the radix-2 Halton sequence from seed value, i
float HashToFloat(uint i)
{    
    return float(i) / float(0xffffffffu);
}

const mat4 kOrderedDither = mat4(vec4(0.0, 8.0, 2.0, 10.), vec4(12., 4., 14., 6.), vec4(3., 11., 1., 9.), vec4(15., 7., 13., 5.));
float OrderedDither()
{    
    return (kOrderedDither[int(gFragCoord.x) & 3][int(gFragCoord.y) & 3] + 1.0) / 17.0;
}

vec3 SampleUnitSphere(vec2 xi)
{
    xi.x = xi.x * 2.0 - 1.0;
    xi.y *= kTwoPi;

    float sinTheta = sqrt(1.0 - xi.x * xi.x);
    return vec3(cos(xi.y) * sinTheta, xi.x, sin(xi.y) * sinTheta);
}

// *******************************************************************************************************
//    Colour functions
// *******************************************************************************************************

vec3 Hue(float phi)
{
    float phiColour = 6.0 * phi;
    int i = int(phiColour);
    vec3 c0 = vec3(((i + 4) / 3) & 1, ((i + 2) / 3) & 1, ((i + 0) / 3) & 1);
    vec3 c1 = vec3(((i + 5) / 3) & 1, ((i + 3) / 3) & 1, ((i + 1) / 3) & 1);             
    return mix(c0, c1, phiColour - float(i));
}

// A Gaussian function that we use to sample the XYZ standard observer 
float CIEXYZGauss(float lambda, float alpha, float mu, float sigma1, float sigma2)
{
   return alpha * exp(sqr(lambda - mu) / (-2.0 * sqr(lambda < mu ? sigma1 : sigma2)));
}

vec3 HSVToRGB(vec3 hsv)
{
    return mix(vec3(0.0), mix(vec3(1.0), Hue(hsv.x), hsv.y), hsv.z);
}

vec3 RGBToHSV( vec3 rgb)
{
    // Value
    vec3 hsv;
    hsv.z = cwiseMax(rgb);

    // Saturation
    float chroma = hsv.z - cwiseMin(rgb);
    hsv.y = (hsv.z < 1e-10) ? 0.0 : (chroma / hsv.z);

    // Hue
    if (chroma < 1e-10)        { hsv.x = 0.0; }
    else if (hsv.z == rgb.x)    { hsv.x = (1.0 / 6.0) * (rgb.y - rgb.z) / chroma; }
    else if (hsv.z == rgb.y)    { hsv.x = (1.0 / 6.0) * (2.0 + (rgb.z - rgb.x) / chroma); }
    else                        { hsv.x = (1.0 / 6.0) * (4.0 + (rgb.x - rgb.y) / chroma); }
    hsv.x = fract(hsv.x + 1.0);

    return hsv;
}

vec3 SampleSpectrum(float lambda)
{
	// Here we use a set of fitted Gaussian curves to approximate the CIE XYZ standard observer.
	// See https://en.wikipedia.org/wiki/CIE_1931_color_space for detals on the formula
	// This allows us to map the sampled wavelength to usable RGB values. This code needs cleaning 
	// up because we do an unnecessary normalisation steps as we map from lambda to XYZ to RGB.

	#define kRNorm (7000.0 - 3800.0) / 1143.07
	#define kGNorm (7000.0 - 3800.0) / 1068.7
	#define kBNorm (7000.0 - 3800.0) / 1068.25

	// Sample the Gaussian approximations
	vec3 xyz;
	xyz.x = (CIEXYZGauss(lambda, 1.056, 5998.0, 379.0, 310.0) +
             CIEXYZGauss(lambda, 0.362, 4420.0, 160.0, 267.0) +
             CIEXYZGauss(lambda, 0.065, 5011.0, 204.0, 262.0)) * kRNorm;
	xyz.y = (CIEXYZGauss(lambda, 0.821, 5688.0, 469.0, 405.0) +
             CIEXYZGauss(lambda, 0.286, 5309.0, 163.0, 311.0)) * kGNorm;
	xyz.z = (CIEXYZGauss(lambda, 1.217, 4370.0, 118.0, 360.0) +
             CIEXYZGauss(lambda, 0.681, 4590.0, 260.0, 138.0)) * kBNorm;

	// XYZ to RGB linear transform
	vec3 rgb;
	rgb.r = (2.04159 * xyz.x - 0.5650 * xyz.y - 0.34473 * xyz.z) / (2.0 * 0.565);
	rgb.g = (-0.96924 * xyz.x + 1.87596 * xyz.y + 0.04155 * xyz.z) / (2.0 * 0.472);
	rgb.b = (0.01344 * xyz.x - 0.11863 * xyz.y + 1.01517 * xyz.z) / (2.0 * 0.452);

	return rgb;
}

// *******************************************************************************************************
//    Ray tracing
// *******************************************************************************************************

#define kInvalidHit              -1.0
#define kNullRay                 -1.0

struct Transform
{
    vec3 trans;
    mat3 rot;
    float sca;
};

mat3 Identity()
{
    return mat3(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));
}

mat3 ScaleMat3(float scale)
{
    float invScale = 1.0f / scale;
	return mat3(vec3(invScale, 0.0, 0.0),
			vec3(0.0, invScale, 0.0),
			vec3(0.0, 0.0, invScale));
}

mat3 RotXMat3(float theta)
{
    float cosTheta = cos(theta), sinTheta = sin(theta);
	return mat3(vec3(1.0, 0.0, 0.0),
			vec3(0.0, cosTheta, -sinTheta),
			vec3(0.0, sinTheta, cosTheta));
}

mat3 RotYMat3(const float theta)
{
    float cosTheta = cos(theta), sinTheta = sin(theta);
	return mat3(vec3(cosTheta, 0.0, sinTheta),
			vec3(0.0, 1.0, 0.0),
			vec3(-sinTheta, 0.0, cosTheta));
}

mat3 RotZMat3(const float theta)
{
    float cosTheta = cos(theta), sinTheta = sin(theta);
	return mat3(vec3(cosTheta, -sinTheta, 0.0),
			vec3(sinTheta, cosTheta, 0.0),
			vec3(0.0, 0.0, 1.0));
}

Transform CompoundTransform(vec3 trans, vec3 rot, float scale)
{
    Transform transform;
    transform.rot = Identity();
    transform.sca = scale;

    if (rot.x != 0.0) { transform.rot *= RotXMat3(rot.x); }
    if (rot.y != 0.0) { transform.rot *= RotYMat3(rot.y); }
    if (rot.z != 0.0) { transform.rot *= RotZMat3(rot.z); }

    if (scale != 1.0f) { transform.rot *= ScaleMat3(scale); }
    
    return transform;
}

// Fast construction of orthonormal basis using quarternions to avoid expensive normalisation and branching 
// From Duf et al's technical report https://graphics.pixar.com/library/OrthonormalB/paper.pdf, inspired by
// Frisvad's original paper: http://orbit.dtu.dk/files/126824972/onb_frisvad_jgt2012_v2.pdf
mat3 CreateBasis(vec3 n)
{
    float s = sign(n.z);
    float a = -1.0 / (s + n.z);
    float b = n.x * n.y * a;
    
    return mat3(vec3(1.0f + s * n.x * n.x * a, s * b, -s * n.x),
                vec3(b, s + n.y * n.y * a, -n.y),
                n);
}

mat3 CreateBasis(vec3 n, vec3 up)
{
    vec3 tangent = normalize(cross(n, up));
	vec3 cotangent = cross(tangent, n);

	return transpose(mat3(tangent, cotangent, n));
}

// The minimum amount of data required to define an infinite ray in 3D space
struct RayBasic
{
    vec3   o;                   // Origin 
    vec3   d;                   // Direction  
};

// The "full fat" ray objects that most methods will refer to
struct Ray
{
    RayBasic od;                
    float    tNear;             
    vec3     weight;
    uint     flags;
    //float    lambda;    
};

#define kFlagsBackfacing 1u
#define kFlagsSubsurface 2u
#define kFlagsDirectRay  4u

struct HitCtx
{
    vec3     n;
    float    kickoff;    
};

RayBasic RayToObjectSpace(in RayBasic world, in Transform transform) 
{
	RayBasic object;
	object.o = world.o - transform.trans;
	object.d = world.d + object.o;
	object.o = transform.rot * object.o;
	object.d = (transform.rot * object.d) - object.o;
	return object;
}

vec4 SDFSimplexFace(in vec3 p, in vec3 v[4], in vec3 n)
{                             
    for(int i = 0; i < 3; i++)
    {
        vec3 dv = v[(i+1) % 3] - v[i];
        vec3 edgeNorm = normalize(cross(dv, n));
        if(dot(edgeNorm, p - v[i]) > 0.0)
        {            
            float t = clamp((dot(p, dv) - dot(v[i], dv)) / dot(dv, dv), 0.0, 1.0);
            vec3 grad = p - (v[i] + t * dv);
            return vec4(length(grad), grad);
        }
    }
    if(dot(n, p - v[0]) < 0.0) { n = -n; }
    return vec4((dot(p, n) - dot(v[0], n)), n);
}

vec4 SDFDisc(in vec3 p, in vec3 V, in vec3 n, float radius)
{
    // Find the perpendicular point on the plane of the triangle
    vec3 pPlane = p + n * (dot(V, n) - dot(p, n)) - V;
    
    float dist = length(pPlane);
    if(dist > radius) { pPlane *= radius / dist; }    
    pPlane = p - (pPlane + V);
    dist = length(pPlane);
    
    return vec4(dist, pPlane / dist);   
}

vec4 SDFTriangle(in vec3 p, in vec3 V[3], in vec3 n)
{       
    // Localise everything
    V[1] -= V[0]; V[2] -= V[0]; p -= V[0];
    
    // Find the perpendicular point on the plane of the triangle
    vec3 pPlane = p + n * - dot(p, n);
    
    // Project into the most significant principle component
    vec2 v1, v2, q;
    vec3 ext = max(abs(V[1]), abs(V[2]));
    if(ext.x < ext.y)
    {
        if(ext.x < ext.z) { v1 = V[1].yz; v2 = V[2].yz; q = pPlane.yz; }
        else              { v1 = V[1].xy; v2 = V[2].xy; q = pPlane.xy; }
    }
    else
    {
        if(ext.y < ext.z) { v1 = V[1].xz; v2 = V[2].xz; q = pPlane.xz; }
        else              { v1 = V[1].xy; v2 = V[2].xy; q = pPlane.xy; }
    }   
   
    // Calculate barycentric coordinates
    float beta = -(v1.x * q.y - q.x * v1.y) / (v1.y * v2.x - v1.x * v2.y);
    float alpha = (q.y - v2.y * beta) / v1.y;
    
    // Clamp to the edges of the triangle if out of bounds
    if(alpha < 0.0) 
    {        
         pPlane = V[2] * saturate(dot(V[2], pPlane) / dot(V[2], V[2]));
    }
    else if(beta < 0.0)
    {
        pPlane = V[1] * saturate(dot(V[1], pPlane) / dot(V[1], V[1]));
    }
    else if(alpha + beta > 1.0)
    { 
        V[2] -= V[1];
        pPlane = V[1] + V[2] * saturate(dot(V[2], pPlane - V[1]) / dot(V[2], V[2]));
    }
        
    p -= pPlane;
    float pMag = length(p);
    return vec4(pMag, vec3(p.x, p.y, p.z));
}

bool RaySimplexSDF(inout Ray ray, inout vec2 sdfNear, out HitCtx hit, in Transform transform, in float explode, in int interval, in float phase)
{
    #define kSDFMaxSteps             25
    #define kSDFCutoffThreshold      1e-4
    #define kSDFEscapeThreshold      2.0
    #define kSDFRayIncrement         1.0
    #define kSDFFailThreshold        0.1
    
    #define kNumIntervals            16
    #define kClosedInterval          12
    
    RayBasic localRay = RayToObjectSpace(ray.od, transform);
    
    float localMag = length(localRay.d);
    localRay.d /= localMag;
    
    float t = 0.0;
    vec3 p;
    vec4 F;
    bool isSubsurface, isBounded = false;

    const vec3 V[4] = vec3[](vec3(1,1,1)*.5, vec3(-1,-1,1)*.5, vec3(1,-1,-1)*.5, vec3(-1,1,-1)*.5);
    const int I[12] = int[]( 0, 2, 1, 1, 3, 2, 2, 0, 3, 3, 1, 0);
    const vec3 N[4] = vec3[](-vec3(-0.5773502588, 0.5773502588, -0.5773502588), vec3(-0.5773502588, -0.5773502588, -0.5773502588),
                             -vec3(-0.5773502588, -0.5773502588, 0.5773502588), vec3(-0.5773502588, 0.5773502588, 0.5773502588));
                        
    #define kSDFThickness 0.05
    
    float beta = KickDrop(phase, vec2(0.0, 0.0), vec2(0.4, 1.2), vec2(0.5, 1.0));
    float gamma = mix(1.0, 2.5, Impulse(phase, 0.4, 0.15));
    
    if(interval == 0) { explode *= beta; } 
    else if(interval > kClosedInterval) { explode = 0.0; }
    else if(interval == kClosedInterval) { explode *= 1.0 - beta; }
    
    int stepIdx;
    float tNear = ray.tNear * localMag;
    for(stepIdx = 0; stepIdx < kSDFMaxSteps; stepIdx++)
    {
        p = localRay.o + localRay.d * t;
        
        F = vec4(kFltMax);
        for(int j = 0; j < 4; j++)
        {              
            int k = j*3;
            vec3 D = N[j] * 0.5;//mix(V[I[k + interval%3]], V[I[k + (interval+1)%3]], beta) * explode;       
            vec4 FFace = SDFTriangle(p, vec3[3](V[I[j*3]] + D, 
                                                  V[I[j*3+1]] + D, 
                                                  V[I[j*3+2]] + D),
                                                  N[j]);

            FFace.x = FFace.x - kSDFThickness * gamma;
            if(FFace.x < F.x) 
            { 
                F = FFace;                
            }
        }
      
        // On the first iteration, simply determine whether we're inside the isosurface or not
        if(stepIdx == 0) { isSubsurface = F.x < 0.0; }
        // Otherwise, check to see if we're at the surface
        else if(abs(F.x) < kSDFCutoffThreshold) { break; }        
        
        if(!isBounded) { if(F.x < kSDFEscapeThreshold) { isBounded = true; } }
        else if(F.x > kSDFEscapeThreshold) { return false; }
        
        t += (isSubsurface ? -F.x : F.x) * kSDFRayIncrement;
        
        //if(abs(F.x / localMag) < sdfNear.x)
        //            sdfNear = vec2(abs(F.x / localMag), t / localMag);
        
        if(t > tNear) { break; }
        
        p = localRay.o + t * localRay.d;
    }        
    
    if(F.x > kSDFFailThreshold || t > tNear) { return false; }
    
    ray.tNear = t / localMag;
    hit.n = transpose(transform.rot) * normalize(F.yzw) * transform.sca;
    ray.flags = (ray.flags & ~kFlagsBackfacing) | ((isSubsurface) ? kFlagsBackfacing : 0u);
    hit.kickoff = 1e-2;

    return true;
}

Ray CreateOrthographicCameraRay(vec2 uv, vec2 sensorSize, vec3 cameraPos, vec3 cameraLookAt)
{
    vec3 w = normalize(cameraLookAt - cameraPos);
    mat3 basis = CreateBasis(w, vec3(0.0, 1.0, 0.0));    
    
    Ray ray;
    ray.od.o = cameraPos + vec3(uv * sensorSize, 0.0) * basis;
    ray.od.d = w;
    ray.weight = vec3(1.0);
    return ray;
}

vec3 RayPointAt(in Ray ray, float t)
{
    return ray.od.o + ray.od.d * t;
}

// *******************************************************************************************************
//    Filters
// *******************************************************************************************************

#define kApplyBloom               true
#define kBloomGain                1.5             // The strength of the bloom effect 
#define kBloomTint                vec3(1.0)       // The tint applied to the bloom effect
#define kBloomWidth               (0.1 / float(kScreenDownsample))             // The width of the bloom effect as a proportion of the buffer width
#define kBloomHeight              (0.1 / float(kScreenDownsample))             // The height of the bloom effect as a proportion of the buffer height
#define kBloomShape               0.5             // The fall-off of the bloom shape. Higher value = steeper fall-off
#define kBloomDownsample          2               // How much the bloom buffer is downsampled. Higher value = lower quality, but faster
#define kDebugBloom               false           // Show only the bloom in the final comp
#define kBloomBurnIn              vec3(0.3)     

// Seperable bloom function. This filter requires two passes in the horizontal and vertical directions which are combined as a post-process
// effect after each frame. The accuracy/cost of the effect can be tuned by dialing the kBloomDownsample parameter. 
vec3 Bloom(vec2 fragCoord, vec3 iResolution, ivec2 delta, sampler2D renderSampler)
{        
    vec2 scaledResolution = vec2(iResolution.x, iResolution.y) / float((delta.x == 1) ? kBloomDownsample : 1);
   
    if(fragCoord.x > scaledResolution.x || fragCoord.y > scaledResolution.y) { return kZero; }
    
    float bloomSize = (delta.x == 1) ? kBloomWidth : kBloomHeight;
    
    int kKernelWidth = int(bloomSize * max(iResolution.x, iResolution.y) + 0.5) / ((delta.x == 1) ? kBloomDownsample : 1);
    vec3 sumWeights = vec3(0.0);
    vec3 sumRgb = vec3(0.0);
    for(int i = -kKernelWidth; i <= kKernelWidth; i++)
    {      
        vec2 xy = vec2(fragCoord.x + float(i * delta.x), fragCoord.y + float(i * delta.y));
        
        if(delta.x == 1) { xy *= float(kBloomDownsample); }
        else { xy /= float(kBloomDownsample); }
        
        if(xy.x < 0.0 || xy.x > iResolution.x || xy.y < 0.0 || xy.y > iResolution.y) { continue; }
            
        vec4 texel = texture(renderSampler, xy / iResolution.xy);
        vec3 rgb = max(texel.xyz / max(1.0, texel.w), vec3(0.0));            
        float d = float(abs(i)) / float(kKernelWidth);
           
        vec3 weight = kOne;
        if(i != 0)
        {
            // Currently using a single weight although this effect can be done per-channel
            float kernel = pow(max(0.0, (exp(-sqr(d * 4.0)) - 0.0183156) / 0.981684), kBloomShape);            
            weight = kOne * kernel;
        }
            
        sumRgb += ((delta.y == 1) ? rgb : max(kZero, rgb - kBloomBurnIn)) * weight;         
        sumWeights += weight;
    }
    
    sumRgb = sumRgb / sumWeights;
    
    return (delta.x == 1) ? sumRgb : (sumRgb * kBloomTint);
}