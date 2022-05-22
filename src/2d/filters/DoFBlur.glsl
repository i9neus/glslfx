// #include "HalfFloat.glsl"

float FocalCurve(float depth, float focalPlane, float dof)
{
    return exp(-sqr(depth - focalPlane) / dof);
}

vec4 EvaluateDoFBlur(in vec2 xyScreen, in sampler2D sampler)
{
    #define kDOFBlurRadius 10
    #define kDOFBlurArea ((kDOFBlurRadius*2+1)*(kDOFBlurRadius*2+1))
    #define kDepthGamma 2.
    
    float focalPlane = 5.0 / 3.0;
    float DoF = 0.2;
    
    vec4 texel0 = texelFetch(sampler, ivec2(xy), 0);
    
    float depth0, alpha0, focus0;
    UnpackFloats(texel0.w, depth0, alpha0); 
    depth0 = pow(depth0, kDepthGamma);
    focus0 = FocalCurve(depth0, focalPlane, DoF);
    
    texel0.xyz = saturate(texel0.xyz);

    vec3 sumL = kZero;
    float sumWeights = 0.0;
    float sumAlpha = 0.0;

    for(int v = -kDOFBlurRadius; v <= kDOFBlurRadius; ++v)
    {
        for(int u = -kDOFBlurRadius; u <= kDOFBlurRadius; ++u)
        {            
            ivec2 uv = ivec2(int(xy.x) + u, int(xy.y) + v);
            
            vec4 texelK = texelFetch(sampler, uv, 0);

            float depthK, alphaK, focusK;
            UnpackFloats(texelK.w, depthK, alphaK);
            depthK = pow(depthK, kDepthGamma);
            focusK = FocalCurve(depthK, focalPlane, DoF);
            
            if((u != 0 || v != 0) && alphaK != 0.0 && alphaK != 1.0) { continue; }            
            
            focusK = (depthK < depth0) ? focusK : max(focusK, focus0);
            
            float weight = 0.0;
            float d = length(vec2(u, v));
            float r = max(0.5, float(kDOFBlurRadius) * (1.0 - focusK));
            if(d < r)
            {
                weight = 1.0 / (1.0 + (kPi * r*r));
            }            
   
            sumL += texelK.xyz * weight;
            sumAlpha += alphaK * weight;
            sumWeights += weight;
        }
    }    
    
    sumL /= max(1e-10, float(sumWeights));
    sumAlpha /= max(1e-10, float(sumWeights));
    
    if(sumAlpha > 0.0) { sumL /= sumAlpha; }
    
    return vec4(sumL, sumAlpha);
}