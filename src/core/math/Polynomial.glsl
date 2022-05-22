// #include "Common.glsl"

bool SolveQuadratic(float a, float b, float c, out float t0, out float t1)
{
    float b2ac4 = b * b - 4.0 * a * c;
    if(b2ac4 < 0.0) { return false; } 

    float sqrtb2ac4 = sqrt(b2ac4);
    t0 = (-b + sqrtb2ac4) / (2.0 * a);
    t1 = (-b - sqrtb2ac4) / (2.0 * a);    
    return true;
}

float SolveSingleSolutionCubic(in vec4 s)
{
    float D0 = sqr(s.y) - 3.0*s.x*s.z;
    float D1 = 2.0*cub(s.y) - 9.0*s.x*s.y*s.z + 27.0*sqr(s.x)*s.w;
    
    float C = cubrt((D1 + sqrt(sqr(D1) - 4.0*cub(D0))) * 0.5);
    
    return -1.0 / (3.0 * s.x) * (s.y + C + D0/C);
}

// Solves a cubic equation of the form ax^3 + bx^2 + cx + d == 0. Returns the number of real solutions. 
int SolveCubic(in vec4 s, out vec3 r)
{
    // Not a cubic equation, so try and solve as a quadtratic
    if(abs(s.x) < 1e-10)
    {            
        float c2bd4 = s.z * s.z - 4.0 * s.y * s.w;
        
        // Not a quadratic equation either, so try and solve linearly
        if(c2bd4 < 1e-10) 
        { 
            if(abs(s.z) < 1e-10) { return 0; } // No solutions
            
            r[0] = -s.w / s.z;
            return 1;
        }

        float sqrtc2bd4 = sqrt(c2bd4);            
        r[0] = (-s.z + sqrtc2bd4) / (2.0 * s.y);
        r[1] = (-s.z - sqrtc2bd4) / (2.0 * s.y);      
        return 2;
    }
    
    // Re-express cubic in depressed form
    float p = (3.0*s.x*s.z - s.y*s.y)/(3.0*s.x*s.x);
    float q = (2.0*s.y*s.y*s.y - 9.0*s.x*s.y*s.z + 27.0*s.x*s.x*s.w)/(27.0*s.x*s.x*s.x);    
    float det = 4.0*p*p*p + 27.0*q*q;
    
    // Only one solution
    if(det > 0.0)
    {        
        float alpha = sqrt(q*q/4.0 + p*p*p/27.0);
        float t = cubrt(-q/2.0 + alpha) + cubrt(-q/2.0 - alpha);
        
        r[0] = t - s.y/(3.0*s.x);
        return 1;
    }
    // Three solutions
    else if(det < 0.0)
    {
        float alpha = acos(3.0*q/(2.0*p) * sqrt(-3.0/p)) / 3.0;        
        float beta = 2.0*sqrt(-p/3.0); 
        for(int i = 0; i < 3; i++)
        {
            float t = beta * cos(alpha - 2.0*kPi*float(i)/3.0);
            r[i] = t - s.y/(3.0*s.x);
        }
        return 3;
    } 
    
    return 0;
}


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

float SeriesAntideriv(float a1, float b1, float a2, float b2, float t)
{    
    float alpha = sqr(a1) + sqr(a2);
    float beta = sqrt(sqr(a2*b1 - a1*b2)/alpha);
    return beta * t + 
            (2.0 / 3.0) * cub(0.5 * a1 * b1 + sqr(a1) * t + a2 * (0.5 * b2 + a2 * t)) /
            (sqr(alpha) * beta);
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