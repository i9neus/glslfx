// *******************************************************************************************************
//    2D SVG functions
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