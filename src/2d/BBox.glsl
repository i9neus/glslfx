// *******************************************************************************************************
//    2D bounding box functions
// *******************************************************************************************************

#define BBox vec4

float EdgeLength(in BBox bb, in int dim) 
{ 
	return bb[2 + dim] - bb[dim]; 
}

float MaxEdgeLength(in BBox bb)  
{ 
	return max(EdgeLength(bb, 0), EdgeLength(bb, 1)); 
} 

float MinEdgeLength(in BBox bb) 
{ 
	return min(EdgeLength(bb, 0), EdgeLength(bb, 1)); 
} 

float MeanEdgeLength(in BBox bb) 
{ 
	return (EdgeLength(bb, 0) + EdgeLength(bb, 1)) * 0.5; 
}

float PerimeterLength(in BBox bb) 
{ 
	return 2.0 * (bb.z - bb.x) + 2.0 * (bb.w - bb.y); 
}

BBox Grow(in BBox bb, in float f) 
{ 
	return BBox(bb.xy - f, bb.zw + f); 
}

BBox Shrink(in BBox bb, in float f) 
{ 
	return BBox(bb.xy + f, bb.zw - f); 
}

vec2 Centroid(in BBox bb) 
{ 
	return mix(bb.xy, bb.zw, 0.5); 
}

float Area(in BBox bb) 
{  
	return (bb.z - bb.x) * (bb.w - bb.y); 
}

BBox Union(in BBox a, in BBox b) 
{ 
	return BBox(min(a.x, b.x), min(a.y, b.y), max(a.z, b.z), max(a.w, b.w)); 
}

bool Contains(in BBox bb, in vec2 v) 
{ 
	return v.x >= bb.x && v.y >= bb.y && v.x < bb.z && v.y < bb.w; 
}

bool InPerimeter(in vec4 bb, in vec2 v, float perim)
{
    return Contains(bb, v) && (v.x - bb.x < perim || v.y - bb.y < perim ||
                               bb.z - v.x < perim || bb.w - v.y < perim);
}