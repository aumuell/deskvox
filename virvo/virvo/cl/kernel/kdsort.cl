// Virvo - Virtual Reality Volume Rendering 
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University 
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu 
// 
// This file is part of Virvo. 
// 
// Virvo is free software; you can redistribute it and/or 
// modify it under the terms of the GNU Lesser General Public 
// License as published by the Free Software Foundation; either 
// version 2.1 of the License, or (at your option) any later version. 
// 
// This library is distributed in the hope that it will be useful, 
// but WITHOUT ANY WARRANTY; without even the implied warranty of 
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
// Lesser General Public License for more details. 
// 
// You should have received a copy of the GNU Lesser General Public 
// License along with this library (see license.txt); if not, write to the 
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

__kernel void reorderLeafs(__global uint* indexbuffer,
  __global uint* leafindices, uint numleafs, uint pointsperleaf)
{
  if (get_global_id(0) >= numleafs * pointsperleaf)
  {
    return;
  }

  uint pointidx = get_global_id(0);
  uint localpointidx = pointidx % pointsperleaf;
  uint leafidx = pointidx / pointsperleaf;
  indexbuffer[pointidx] = leafindices[leafidx] * pointsperleaf + localpointidx;
//  indexbuffer[pointidx] = leafidx * pointsperleaf + localpointidx;
}

__kernel void bitonicSort(__global const float4* vertices, __global uint* indexbuffer, __local uint* aux,
  uint iterations, float4 eye)
{
  uint i = get_local_id(0);
  uint wg = get_local_size(0);
  uint offset = get_group_id(0) * wg * iterations;
  indexbuffer += offset;

//  for (uint it = 0; it < 2; ++it)
  {
    aux[i] = indexbuffer[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint len = 1; len < wg; len <<= 1)
    {
      bool direction = ((i & (len << 1)) != 0);
      for (uint inc = len; inc > 0; inc >>=1)
      {
        uint j = i ^ inc;

        uint iidx = indexbuffer[i];
        float3 iv = vertices[iidx].xyz;
        float ikey = length(iv - eye.xyz);

        uint jidx = indexbuffer[j];
        float3 jv = vertices[jidx].xyz;
        float jkey = length(jv - eye.xyz);

        bool lower = (jkey < ikey) || (jkey == ikey && j < i);
        bool swap = lower ^ (j < i) ^ direction;
        barrier(CLK_LOCAL_MEM_FENCE);

        aux[i] = swap ? jidx : iidx;
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }

    indexbuffer[i] = aux[i];

//    indexbuffer += wg;
  }
}

