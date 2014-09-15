#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifndef make_int2
#define make_int2(x, y) ((int2)(x, y))
#endif

#ifndef make_int3
#define make_int3(x, y, z) ((int3)(x, y, z))
#endif

#ifndef make_int4
#define make_int4(x, y, z, w) ((int4)(x, y, z, w))
#endif

#ifndef make_uint2
#define make_uint2(x, y) ((uint2)(x, y))
#endif

#ifndef make_uint3
#define make_uint3(x, y, z) ((uint3)(x, y, z))
#endif

#ifndef make_uint4
#define make_uint4(x, y, z, w) ((uint4)(x, y, z, w))
#endif

#ifndef make_float2
#define make_float2(x, y) ((float2)(x, y))
#endif

#ifndef make_float3
#define make_float3(x, y, z) ((float3)(x, y, z))
#endif

#ifndef make_float4
#define make_float4(x, y, z, w) ((float4)(x, y, z, w))
#endif


typedef struct
{
  int left;
  int bottom;
  int width;
  int height;
} Viewport;

typedef struct
{
  float3 origin;
  float3 direction;
} Ray;

inline Ray make_ray(float3 origin, float3 direction)
{
    Ray result;
    result.origin = origin;
    result.direction = direction;
    return result;
}

typedef struct
{
  float3 min;
  float3 max;
} Aabb;

typedef struct
{
  float m[4][4];
} matrix4x4;

inline float4 mul(__local matrix4x4* m, float4 v)
{
    float4 result;
    result.x = m->m[0][0] * v.x + m->m[1][0] * v.y + m->m[2][0] * v.z + m->m[3][0] * v.w;
    result.y = m->m[0][1] * v.x + m->m[1][1] * v.y + m->m[2][1] * v.z + m->m[3][1] * v.w;
    result.z = m->m[0][2] * v.x + m->m[1][2] * v.y + m->m[2][2] * v.z + m->m[3][2] * v.w;
    result.w = m->m[0][3] * v.x + m->m[1][3] * v.y + m->m[2][3] * v.z + m->m[3][3] * v.w;
    return result;
}

inline float3 perspectiveDivide(float4 v)
{
    float winv = 1.0f / v.w;
    return v.xyz * winv;
}

inline float min4(float a, float b, float c, float d)
{
    return fmin(fmin(fmin(a, b), c), d);
}

inline float max4(float a, float b, float c, float d)
{
    return fmax(fmax(fmax(a, b), c), d);
}

__kernel void trace(
  __global float4* framebuffer,
  __read_only image2d_t bvhnodes, __read_only image2d_t spheres, uint numSpheres,
  __read_only image2d_t scalars, __read_only image2d_t tf,
  __constant matrix4x4* invModelview, __constant matrix4x4* invProjection, Viewport vp,
  __global uint* blockCounter, uint numBlocks, uint gridWidth
)
{
  __local uint blockIndex;
  __local uint blockX;
  __local uint blockY;

  __local matrix4x4 invmv;
  __local matrix4x4 invpr;

  for (;;)
  {
    if (get_local_id(0) == 0 && get_local_id(1) == 0)
    {
      blockIndex = atom_inc(blockCounter);
      blockX = blockIndex % gridWidth;
      blockY = blockIndex / gridWidth;

      for (int i = 0; i < 4; ++i)
      {
        for (int j = 0; j < 4; ++j)
        {
          invmv.m[i][j] = invModelview->m[i][j];
          invpr.m[i][j] = invProjection->m[i][j];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (blockIndex >= numBlocks)
    {
      break;
    }

    uint x = blockX * get_local_size(0) + get_local_id(0);
    uint y = blockY * get_local_size(1) + get_local_id(1);

    if (x >= vp.width || y >= vp.height)
    {
      continue;
    }

    float u = ((float)x / (float)vp.width) * 2.0f - 1.0f;
    float v = ((float)y / (float)vp.height) * 2.0f - 1.0f;

    float4 o = mul(&invmv, mul(&invpr, make_float4(u, v, 0.0f, 1.0f)));
    float4 d = mul(&invmv, mul(&invpr, make_float4(u, v, 1.0f, 1.0f)));
    float3 origin = perspectiveDivide(o);
    float3 direction = perspectiveDivide(d);
    direction = direction - origin;
    direction = normalize(direction);
    Ray ray = make_ray(origin, direction);
    float4 dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

int tmp = 0;
    for (;;)
    {
    float2 t = make_float2(FLT_MAX, FLT_MAX);
    // traverse BVH
#define SENTINEL 0x76543210
    uint primitive = 0;
    uint stack[32];
    stack[0] = SENTINEL;

    char* stackptr = (char*)&stack[0];
    uint nodeaddr = 0;

    float3 invraydir = make_float3(1.0f / ray.direction.x,
        1.0f / ray.direction.y, 1.0f / ray.direction.z);
    float3 minusorioverdir = make_float3(-ray.origin.x / ray.direction.x,
        -ray.origin.y / ray.direction.y, -ray.origin.z / ray.direction.z);

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;
    while (nodeaddr != SENTINEL)
    {
      int2 idx2 = make_int2((nodeaddr * 2) & (MAXTEXWIDTH - 1), (nodeaddr * 2) >> MAXTEXWIDTHLOG2);
      float4 bvh1 = read_imagef(bvhnodes, sampler, idx2);
      idx2 = make_int2((nodeaddr * 2 + 1) & (MAXTEXWIDTH - 1), (nodeaddr * 2 + 1) >> MAXTEXWIDTHLOG2);
      float4 bvh2 = read_imagef(bvhnodes, sampler, idx2);
      float tx1 = bvh1.x * invraydir.x + minusorioverdir.x;
      float tx2 = bvh2.x * invraydir.x + minusorioverdir.x;
      float ty1 = bvh1.y * invraydir.y + minusorioverdir.y;
      float ty2 = bvh2.y * invraydir.y + minusorioverdir.y;
      float tz1 = bvh1.z * invraydir.z + minusorioverdir.z;
      float tz2 = bvh2.z * invraydir.z + minusorioverdir.z;
      float tmin = max4(fmin(tx1, tx2), fmin(ty1, ty2), fmin(tz1, tz2), FLT_EPSILON);
      float tmax = min4(fmax(tx1, tx2), fmax(ty1, ty2), fmax(tz1, tz2), t.y);

      if (tmax >= tmin)
      {
        bool leaf = bvh2.w > FLT_EPSILON;
        if (!leaf)
        {
          uint firstChildOffset = bvh1.w;
          stackptr += 4;
          *(uint*)stackptr = firstChildOffset + 1;
          nodeaddr = firstChildOffset;
          continue;
        }
        else
        {
          uint firstSphereOffset = bvh1.w;
          for (uint i = firstSphereOffset; i < firstSphereOffset + numSpheres; ++i)
          {
            int2 sidx2 = make_int2(i & (MAXTEXWIDTH - 1), i >> MAXTEXWIDTHLOG2);
            float4 sphere = read_imagef(spheres, sampler, sidx2);
            float radiusSqr = sphere.w * sphere.w;
            Ray r = ray;
            r.origin -= sphere.xyz;
            float A = r.direction.x * r.direction.x + r.direction.y * r.direction.y + r.direction.z * r.direction.z;
            float B = 2 * (r.direction.x * r.origin.x + r.direction.y * r.origin.y + r.direction.z * r.origin.z);
            float C = r.origin.x * r.origin.x + r.origin.y * r.origin.y + r.origin.z * r.origin.z - radiusSqr;

            float tnear = 0.0f;
            float tfar = 0.0f;
            float discrim = B * B - 4.0f * A * C;
            if (discrim < 0.0f)
            {
              tnear = -1.0f;
              tfar = -1.0f;
              continue;
            }
            float rootDiscrim = sqrt(discrim);
            float q;
            if (B < 0)
            {
              q = -0.5f * (B - rootDiscrim);
            }
            else
            {
              q = -0.5f * (B + rootDiscrim);
            }
            tnear = q / A;
            tfar = C / q;
            if (tnear > tfar)
            {
              float tmp = tnear;
              tnear = tfar;
              tfar = tmp;
              if (tnear < t.x)
              {
                t.x = tnear;
                t.y = tfar;
                primitive = i;
              }
            }
          }
        }
      }

      nodeaddr = *(uint*)stackptr;
      stackptr -= 4;
    }

    if (t.x < FLT_MAX)
          {
            int2 sidx2 = make_int2(primitive & (MAXTEXWIDTH - 1), primitive >> MAXTEXWIDTHLOG2);
            float4 sphere = read_imagef(spheres, sampler, sidx2);

            float dist = t.y - t.x;
            float radius2 = sphere.w + sphere.w;
            dist /= radius2;

            float4 vol = read_imagef(scalars, sampler, sidx2);
            float2 texcoord = make_float2(vol.x, 1.0f);
            sampler_t sampler2 = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP_TO_EDGE;
            float4 src = read_imagef(tf, sampler2, texcoord);

            if (src.w > 1.0f) src.w = 1.0f;
            src.w *= dist;
            src.x *= src.w;
            src.y *= src.w;
            src.z *= src.w;
            dst = dst + src * (1.0f - dst.w);

            float3 hitpoint = ray.origin + t.y * ray.direction;
            hitpoint = hitpoint + FLT_EPSILON * ray.direction;
            ray = make_ray(hitpoint, ray.direction);
          }

#undef SENTINEL

    if (t.x == FLT_MAX || dst.w > 0.9 || tmp > 1000)
    {
      break;
    }++tmp;
    }
    framebuffer[y * vp.width + x] = dst;
  }
}
