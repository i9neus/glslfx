#define kArcSplineSize 8
const vec3 kArcSpline[kArcSplineSize] = vec3[kArcSplineSize]
(
    vec3(0.08713340291649768, 0.7071067811865476, 0.3535533905932738),
    vec3(0.27460414208944717, 0.4472135954999581, 0.4472135954999579),
    vec3(0.562285023557882, 0.24253562503633397, 0.4850712500726658),
    vec3(0.8920972020050328, 0.12403473458920722, 0.49613893835683387),
    vec3(1.2343292297009185, 0.06237828615518204, 0.49902628924144427),
    vec3(1.5798075292695444, 0.031234752377724107, 0.49975603804353935),
    vec3(1.9261066707274495, 0.015623093000534993, 0.4999389760173476),
    vec3(2.2726116095509497, 0.007812261592334835, 0.4999847419093939)
);

float InvSpiralArcLength(float arc)
{
    if(arc <= 0.0) { return 0.0; }
    float p = log2(1.0 + arc);
    
    int i;
    float a;    
    if(p >= float(kArcSplineSize - 1)) { i = kArcSplineSize - 2; a = 1.0; }
    else { i = int(p); a = p - float(i); }    
    
    vec3 abc = mix(kArcSpline[i], kArcSpline[i+1], a);
    abc.x -= arc;
    
    return (-abc.y + sqrt(abc.y * abc.y - 4.0 * abc.z * abc.x)) / (2.0 * abc.z);    
}

float SpiralArcLength(float theta)
{
    return 0.5 * (theta * sqrt(1.0 + theta * theta) + log(theta + sqrt(1.0 + theta * theta)));
}