vec4 ToggleKeystate(in ivec2 xy, in int keyCode, in int defaultValue)
{
    int keyState = int(texelFetch(iChannel2, ivec2(keyCode, 0), 0).x != 0.0);    
    ivec4 toggleState = ivec4(texelFetch(iChannel0, ivec2(xy.x, kAttrRow), 0));
    if(toggleState.x == KEY_UNDEFINED) { toggleState.x = defaultValue; }
    
    if(keyState != toggleState.y) { toggleState.x = toggleState.x % 4 + 1; }
    toggleState.y = keyState;
     
    return vec4(toggleState);
}