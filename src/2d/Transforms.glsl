// *******************************************************************************************************
//    2D spatial transformations 
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

// Returns the polar distance r to the perimeter of an n-sided polygon
float DrosteNgon(in float phi, in float offset)
{
    #define kDrosteBlades 5
    #define kDrosteBladeCurvature 2.5

    float piBlades = kPi / kDrosteBlades;
    float bladeRadius = cos(piBlades) / cos(mod(((phi + offset) + piBlades) + piBlades, 2.0f*piBlades) - piBlades);
    
    // Take into account the blade curvature
    return kDrosteBladeCurvature + bladeRadius * (1.0 - kDrosteBladeCurvature);
}    

/** Remaps the xy input in the range [0, 1] according to the Droste spiral
Output: 
 x: u coordinate [0, 1]
 y: v coordinate [0, 1]
 z: The number of turns into the spiral
 w: The normalised angle to the origin in the range [0, 1]
**/
vec4 DrosteMap(in vec2 xy, in float time)
{
    #define kDrosteZoomSpeed 0.2

    // Remap into range [-1, 1]
    float x = ((1. - xy.x) * 2. - 1.);
    float y = ((xy.y) * 2. - 1.);
    
    // Just using constants for now
    float Exp = kExp;
    float LogExp = log(Exp);

    float phi = atan(y, x);
    float chi = mod(kPi + phi / (2.0*kPi) + 0.25, 1.0) + mod(kDrosteZoomSpeed * time, 1.0);
    float r = pow(Exp, -chi) * sqrt(x*x + y*y) / kRoot2;
    float s = r / DrosteNgon(phi, time);
    float turn = ceil(log(s) / LogExp);
    float alpha = r / pow(Exp, turn);
   
    vec2 uv;
    uv.x = (alpha * cos(phi) + 1.0) * 0.5;
    uv.y = (alpha * sin(phi) + 1.0) * 0.5;   
    return vec4(uv, turn, chi);
}
