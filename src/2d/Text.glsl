float Character(int code, vec2 uv, bool invert)
{      
    if(uv.x < 0.0 || uv.x >= 1.0 || uv.y < 0.0 || uv.y >= 1.0) { return 0.0; }
    
    #define kNumeralXMargin 0.28
    #define kNumeralYMargin 0.0
    uv.x = kNumeralXMargin + (1.0 - 2.0 * kNumeralXMargin) * uv.x;
    uv.y = kNumeralYMargin + (1.0 - 2.0 * kNumeralYMargin) * uv.y;
    
    uv.x = 64.0 * (float(code % 16) + uv.x) / 1024.0;
    uv.y = 64.0 * (float(15 - code / 16) + uv.y) / 1024.0;
    
    return invert ? (1.0 - texture(iChannel2, uv, 0.).x) : texture(iChannel2, uv, 0.).x;
}

#define RenderText(str, strLen, xy, pos, textSize, result, invert)\
    vec2 cursor = (xy - pos + vec2(textSize.x * float(strLen) / 2.0, textSize / 2.0)) / textSize;\
    if(cursor.x >= 0.0 && int(cursor.x) < strLen && cursor.y >= 0.0 && cursor.y < 1.0)\
    {\
        vec2 delta = fract(cursor);\
        result += Character(str[int(cursor.x)], delta, invert);\
    }\
    else { result = -1.0; }\
    
void RenderDigit(vec2 xy, float n, int dp, vec2 textPos, vec2 textSize, inout vec4 rgba)
{        
    #define kMaxTextLength 10
    int[kMaxTextLength] str;  
    
    int numDigits = 0;
    int charIdx = 0;
    bool isNeg = false;

    if(n < 0.0)
    {
        isNeg = true;
        n = abs(n);
        numDigits++;
    }

    int intPart = int(n);
    if(dp > 0)
    {
        intPart = int(n * pow(10.0, float(dp)));
        numDigits += 1 + dp;
    }
    
    numDigits = min(10, numDigits + 1 + int(log(1.0 + n) / kLog10));
    for(int i = numDigits - 1, j = 1; i >= 0; j *= 10) 
    { 
        str[i--] = 48 + (intPart / j) % 10;
        if(dp > 0 && numDigits - 1 - i == dp)
        {
            str[i--] = 46;
        }        
    } 
    
    if(isNeg) { str[0] = 45; }
    
    float glyph = 0.0;
    RenderText(str, numDigits, xy, textPos, textSize, glyph, false);
    if(glyph > 0.0)
    {
        rgba = mix(rgba, vec4(1.0), glyph);
        return;        
    }    
}