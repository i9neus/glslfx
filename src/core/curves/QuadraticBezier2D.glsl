// Calculates the closest point on the quadratic curve to position xy
bool QuadradicSplinePerpendicularPoint(in vec3 abcX, in vec3 abcY, in vec2 xy, out float tPerp, out vec2 xyPerp)
{     
    float a0 = abcX.x, b0 = abcX.y, c0 = abcX.z;
    float d0 = abcY.x, e0 = abcY.y, f0 = abcY.z;
    float n0 = xy.x, m0 = xy.y;   
    
    float a = -2.0*a0*a0 - 2.0*d0*d0;
    float b = -3.0*a0*b0 - 3.0*d0*e0;
    float c = -b0*b0 - 2.0*a0*c0 - e0*e0 - 2.0*d0*f0 + 2.0*d0*m0 + 2.0*a0*n0;
    float d = -b0*c0 - e0*f0 + e0*m0 +  b0*n0;
    
    vec3 solutions;
    int numSolutions = SolveCubic(vec4(a, b, c, d), solutions);   
    
    if(numSolutions == 0) { return false; }
    
    xyPerp = vec2(kFltMax);
    float nearest = kFltMax;
    for(int idx = 0; idx < numSolutions; ++idx)
    {
        float t = saturate(solutions[idx]);
        vec2 perp = vec2(abcX.x * t*t + abcX.y * t + abcX.z,
                        abcY.x * t*t + abcY.y * t + abcY.z);                        
                        
        float dist = length2(xy - perp);
        if(dist < nearest)
        {
            nearest = dist;
            xyPerp = perp;
            tPerp = t;
        }            
    }
    
    return true;
}

// Returns the 2D bounding box of the spline in the lower and upper parts of a vec4    
vec4 GetQuadraticSplineBoundingBox(in vec3 abc[2])
{
    vec4 bb;    
    for(int d = 0; d < 2; ++d)
    {
        // Min/max 
        float extremum = abc[d].x + abc[d].y + abc[d].z;
        bb[d] = min(abc[d].z, extremum);
        bb[d+2] = max(abc[d].z, extremum);
    
        if(abs(abc[d].x) > 1e-10)
        {
            float t = -abc[d].y / (2.0 * abc[d].x);
            if(t >= 0.0 && t <= 1.0)
            {            
                float inflect = abc[d].x*t*t + abc[d].y*t + abc[d].z;
                bb[d] = min(bb[d], inflect);
                bb[d+2] = max(bb[d+2], inflect);       
            }
        }
    }
    
    return bb;
}

float Arc(const float a1, const float b1, const float a2, const float b2, const float t)
{
	return sqrt(sqr(b1)+sqr(b2) + (a1*b1 + a2*b2)*4.0*t + (sqr(a1) + sqr(a2))*4.0*sqr(t));
}

float DArc(const float a1, const float b1, const float a2, const float b2, const float t)
{
	return (((a1*b1 + a2*b2)*2.0 + (sqr(a1) + sqr(a2))*4.0*t)) / Arc(a1, b1, a2, b2, t);
}

vec4 ApproxQuadraticSplineCDF(in vec3 abcX, in vec3 abcY)
{
    float a1 = abcX.x, b1 = abcX.y;
    float a2 = abcY.x, b2 = abcY.y;    
    
    float tMin = clamp((-a1*b1 - a2*b2)/(2.0 * (a1*a1 + a2*a2)), 0.15, 1.0 - 0.15);
    float tMinGrad = DArc(a1, b1, a2, b2, tMin);   
    
    float tFit = ((tMin < 0.5) ? 1.0 : 0.0);
    
    mat3 M = inverse(mat3(vec3(sqr(tFit), tFit, 1.0), 
                          vec3(sqr(tMin), tMin, 1.0), 
                          vec3(2.0*tMin, 1.0, 0.0)));
                          
    vec3 B = vec3(Arc(a1, b1, a2, b2, tFit),
                  Arc(a1, b1, a2, b2, tMin),
                  tMinGrad);
                
    vec3 fit = B*M;
    return vec4(fit.x / 3.0, fit.y / 2.0, fit.z, 0.0);
}

float SampleArc(in vec3 abcX, in vec3 abcY, in float xi)
{
    vec4 n = ApproxQuadraticSplineCDF(abcX, abcY);
    
    n.w = -(n.x + n.y + n.z) * xi;
    
    float t= saturate(SolveSingleSolutionCubic(n));
    return saturate(mix(xi, t, 5.0 / 7.0));
}

float InvSampleArc(in vec3 abcX, in vec3 abcY, in float xi)
{
    vec4 n = ApproxQuadraticSplineCDF(abcX, abcY);
    
    float t = (xi * (xi * ((xi * n.x) + n.y) + n.z) + n.w) / (n.x + n.y + n.z);   
    return saturate(mix(t, xi, 0.2));
}

vec2 EvaluateArc(in vec3 abcX, in vec3 abcY, in float t)
{
    return vec2(abcX.x * t*t + abcX.y * t + abcX.z, abcY.x * t*t + abcY.y * t + abcY.z); 
}

void EquiangularSampleCurve(in float L0, in float L1, in float D, in float xi, out float tEqui, out float pdf)
{
    float cdf0 = CauchyCDF(L0, D);
    float cdf1 = CauchyCDF(L1, D);
    float x = tan((mix(cdf0, cdf1, xi) - 0.5) * kPi) * D;
        
    tEqui = (x - L0) / (L1 - L0);
    pdf = CauchyPDF(x, D) / max(1e-10, cdf1 - cdf0);
}

// Reduces two linear curves, C0 and C1, to a single curve using principle components analysis
vec4 DecimateLinearCurves(vec4 C0, vec4 C1)
{
    // Compute the mean of the curves
    vec2 mean = vec2(0.0);
    mean.x = (C0.x + C0.z + C1.x + C1.z) * 0.25;
    mean.y = (C0.y + C0.w + C1.y + C1.w) * 0.25;
    
    // Construct the covariance matrix
    mat2 covM;
    for(int i = 0; i < 2; ++i)
    {
        for(int j = 0; j < 2; ++j)
        {
            covM[i][j] = ((C0[i] - mean[i]) * (C0[j] - mean[j]) + 
                         (C0[i+2] - mean[i]) * (C0[j+2] - mean[j]) +
                         (C1[i] - mean[i]) *   (C1[j] - mean[j]) + 
                         (C1[i+2] - mean[i]) * (C1[j+2] - mean[j])) * 0.25;
                         
        }
    }    
    
    // Decompose it to extract the eigenvectors and eigenvalues
    vec2 v0, v1;
    float l0, l1;
    if(!Eigendecompose(covM, v0, v1, l0, l1)) { return vec4(0.0); }
    
    // The projection basis is the largest principle component 
    vec2 basis = (l0 > l1) ? v0 : v1;    

    // Project each point into the basis and find its extremes
    float tMin = kFltMax, tMax = -kFltMax;
    for(int i = 0; i < 4; ++i) 
    { 
        vec2 f = (i < 2) ? ((i%2 == 0) ? C0.xy : C0.zw) : ((i%2 == 0) ? C1.xy : C1.zw); 
        float t = dot(f, basis) - dot(mean, basis);
        tMin = min(tMin, t);
        tMax = max(tMax, t);
     }
     
     // The extrema of the projected points form the new curve    
     return vec4(mean + basis * tMin, mean + basis * tMax);
}