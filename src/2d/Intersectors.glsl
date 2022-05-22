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