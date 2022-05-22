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
    return float(HashOf(i)) / float(0xffffffffu);
}

vec3 SampleUnitSphere(vec2 xi)
{
    xi.x = xi.x * 2.0 - 1.0;
    xi.y *= kTwoPi;

    float sinTheta = sqrt(1.0 - xi.x * xi.x);
    return vec3(cos(xi.y) * sinTheta, xi.x, sin(xi.y) * sinTheta);
}