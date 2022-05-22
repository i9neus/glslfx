// Basic ray position and direction
struct RayBasic
{
    vec3   o;                   // Origin 
    vec3   d;                   // Direction  
};

// The "full fat" ray objects that most methods will refer to
struct Ray
{
    RayBasic od;   
    
    float    tNear;
    vec3     weight;
    float    pdf;
    uint     flags;
};

struct HitCtx
{
    vec3     n;
    vec2     uv;
    float    kickoff;
    int      matID;
};

#define kFlagsBackfacing      1u
#define kFlagsSubsurface      2u
#define kFlagsDirectRay       4u
#define kFlagsScattered       8u
#define kFlagsLightSample     16u
#define kFlagsProbePath       32u
#define kFlagsCausticPath     64u
#define kFlagsVolumetricPath  128u

//#define InheritFlags(ray) (ray.flags & kFlagsScattered)
#define InheritFlags(ray) (ray.flags & (kFlagsProbePath | kFlagsCausticPath))

#define PointAt(ray) (ray.od.o + ray.od.d * ray.tNear)
#define PointAt(ray, t) (ray.od.o + ray.od.d * t)

#define IsBackfacing(ray) ((ray.flags & kFlagsBackfacing) != 0u)
#define IsSubsurface(ray) ((ray.flags & kFlagsSubsurface) != 0u)
#define IsScattered(ray) ((ray.flags & kFlagsScattered) != 0u)
#define IsLightSample(ray) ((ray.flags & kFlagsLightSample) != 0u)
#define IsProbePath(ray) ((ray.flags & kFlagsProbePath) != 0u)
#define IsCausticPath(ray) ((ray.flags & kFlagsCausticPath) != 0u)
#define IsVolumetricPath(ray) ((ray.flags & kFlagsVolumetricPath) != 0u)

void SetRayFlag(inout Ray ray, in uint flag, in bool set)
{
    ray.flags &= ~flag;
    if(set) 
    {
        ray.flags |= flag;
    }  
}

void CreateRay(inout Ray ray, vec3 o, vec3 d, vec3 kickoff, vec3 weight, float pdf, uint flags)
{     
    ray.od.o = o + kickoff;
    ray.od.d = d;
    ray.tNear = kFltMax;
    ray.weight = weight;
    ray.pdf = pdf;
    ray.flags = flags;
}

RayBasic RayToObjectSpace(in RayBasic world, in Transform transform) 
{
    RayBasic object;
    object.o = world.o - transform.trans;
    object.d = world.d + object.o;
    object.o = transform.rot * object.o;
    object.d = (transform.rot * object.d) - object.o;
    return object;
}