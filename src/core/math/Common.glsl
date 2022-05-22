// #include "Constants.glsl"

float cubrt(float a)           { return sign(a) * pow(abs(a), 1.0 / 3.0); }
float toRad(float deg)         { return kTwoPi * deg / 360.0; }
float toDeg(float rad)         { return 360.0 * rad / kTwoPi; }
float sqr(float a)             { return a * a; }
vec2 sqr(vec2 a)               { return a * a; }
vec3 sqr(vec3 a)               { return a * a; }
vec4 sqr(vec4 a)               { return a * a; }
int sqr(int a)                 { return a * a; }
int cub(int a)                 { return a * a * a; }
float cub(float a)             { return a * a * a; }
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
vec3 saturate(vec3 a)          { return clamp(a, 0.0, 1.0); }
vec4 saturate(vec4 a)          { return clamp(a, 0.0, 1.0); }
float saw01(float a)           { return abs(fract(a) * 2.0 - 1.0); }
float cwiseMax(vec3 v)         { return (v.x > v.y) ? ((v.x > v.z) ? v.x : v.z) : ((v.y > v.z) ? v.y : v.z); }
float cwiseMax(vec2 v)         { return (v.x > v.y) ? v.x : v.y; }
int cwiseMax(ivec2 v)        { return (v.x > v.y) ? v.x : v.y; }
float cwiseMin(vec3 v)         { return (v.x < v.y) ? ((v.x < v.z) ? v.x : v.z) : ((v.y < v.z) ? v.y : v.z); }
float cwiseMin(vec2 v)         { return (v.x < v.y) ? v.x : v.y; }
float max3(float a, float b, float c) { return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c); }
float min3(float a, float b, float c) { return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c); }
void sort(inout float a, inout float b) { if(a > b) { float s = a; a = b; b = s; } }
void swap(inout float a, inout float b) { float s = a; a = b; b = s; }
void swap(inout int a, inout int b) { int s = a; a = b; b = s; }

vec3 safeAtan(vec3 a, vec3 b)
{
    vec3 r;
    #define kAtanEpsilon 1e-10
    r.x = (abs(a.x) < kAtanEpsilon && abs(b.x) < kAtanEpsilon) ? 0.0 : atan(a.x, b.x); 
    r.y = (abs(a.y) < kAtanEpsilon && abs(b.y) < kAtanEpsilon) ? 0.0 : atan(a.y, b.y); 
    r.z = (abs(a.z) < kAtanEpsilon && abs(b.z) < kAtanEpsilon) ? 0.0 : atan(a.z, b.z); 
    return r;
}

vec3 GuardedNormalise(vec3 v, vec3 n)
{
    float len = length(v);
    return (len > 1e-10) ? (v / len) : n;
}

vec3 SafeNormaliseTexel(vec4 t)
{
    return t.xyz / max(1e-15, t.w);
}

vec4 Sign(vec4 v)
{
    return step(vec4(0.0), v) * 2.0 - 1.0;
}

float Sign(float v)
{
    return step(0.0, v) * 2.0 - 1.0;
}

bool IsNan( float val )
{
    return ( val < 0.0 || 0.0 < val || val == 0.0 ) ? false : true;
}

bvec3 IsNan( vec3 val )
{
    return bvec3( ( val.x < 0.0 || 0.0 < val.x || val.x == 0.0 ) ? false : true, 
                  ( val.y < 0.0 || 0.0 < val.y || val.y == 0.0 ) ? false : true, 
                  ( val.z < 0.0 || 0.0 < val.z || val.z == 0.0 ) ? false : true);
}

bvec4 IsNan( vec4 val )
{
    return bvec4( ( val.x < 0.0 || 0.0 < val.x || val.x == 0.0 ) ? false : true, 
                  ( val.y < 0.0 || 0.0 < val.y || val.y == 0.0 ) ? false : true, 
                  ( val.z < 0.0 || 0.0 < val.z || val.z == 0.0 ) ? false : true,
                  ( val.w < 0.0 || 0.0 < val.w || val.w == 0.0 ) ? false : true);
}

float UintToFloat(uint i)
{    
    return float(i & ((1u << 30u) - 1u)) / float((1u << 30u) - 1u);
}


#define SignedGamma(v, gamma) (sign(v) * pow(abs(v), gamma))