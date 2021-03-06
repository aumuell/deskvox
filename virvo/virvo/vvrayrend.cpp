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


#include "vvrayrend.h"


#ifdef HAVE_CUDA


#include "vvrayrend-common.h"

#include <GL/glew.h>

#include "cuda/array.h"
#include "cuda/memory.h"
#include "cuda/symbol.h"
#include "cuda/texture.h"
#include "cuda/utils.h"

#include "gl/util.h"

#include "math/math.h"

#include "texture/texture.h"

#include "vvcudaimg.h"
#include "vvdebugmsg.h"
#include "vvvoldesc.h"

#include "private/vvlog.h"

namespace cu = virvo::cuda;
namespace gl = virvo::gl;

using virvo::mat4;
using virvo::vec3f;
using virvo::vec4;


// in vvrayrend.cu
extern cu::Texture cVolTexture8;
extern cu::Texture cVolTexture16;
extern cu::Texture cPrefilterTexture;
extern cu::Texture cTFTexture;

// in vvrayrend.cu
extern cu::Symbol<matrix4x4> cInvViewMatrix;
extern cu::Symbol<matrix4x4> cMvPrMatrix;

// in vvrayrend.cu
extern "C" void CallRayRendKernel(const RayRendKernelParams& params,
                                  uchar4* d_output, const uint width, const uint height,
                                  const uint texwidth, const float dist,
                                  const float3 volPos, const float3 volSizeHalf,
                                  const float3 probePos, const float3 probeSizeHalf,
                                  float3 texsize,
                                  virvo::tex_filter_mode filter_mode,
                                  const float3 Lpos, const float3 V,
                                  float constAtt, float linearAtt, float quadAtt,
                                  const bool clipPlane,
                                  const bool clipSphere,
                                  const bool useSphereAsProbe,
                                  const float3 sphereCenter, const float sphereRadius,
                                  const float3 planeNormal, const float planeDist,
                                  void* d_depth, int dp,
                                  const float2 ibrPlanes,
                                  const IbrMode ibrMode,
                                  bool twoPassIbr);


IbrMode getIbrMode(vvRenderer::IbrMode mode)
{
  switch (mode)
  {
  case vvRenderer::VV_ENTRANCE:
    return VV_ENTRANCE;
  case vvRenderer::VV_EXIT:
    return VV_EXIT;
  case vvRenderer::VV_MIDPOINT:
    return VV_MIDPOINT;
  case vvRenderer::VV_THRESHOLD:
    return VV_THRESHOLD;
  case vvRenderer::VV_PEAK:
    return VV_PEAK;
  case vvRenderer::VV_GRADIENT:
    return VV_GRADIENT;
  case VV_REL_THRESHOLD:
    return VV_REL_THRESHOLD;
  case VV_EN_EX_MEAN:
    return VV_EN_EX_MEAN;
  default:
    return VV_NONE;
  }
}


struct vvRayRend::Impl
{
private:
  cu::Array* volumeArrays;
  size_t numVolumeArrays;
public:
  cu::Array transferFuncArray;
  cudaChannelFormatDesc channelDesc;

  Impl()
    : volumeArrays(0)
    , numVolumeArrays(0)
    , transferFuncArray()
    , channelDesc()
  {
  }

  ~Impl()
  {
    delete [] volumeArrays;
  }

  cu::Array& getVolumeArray(size_t index) const
  {
    assert( index < numVolumeArrays );
    return volumeArrays[index];
  }

  void allocateVolumeArrays(size_t size)
  {
    // Create a new array
    cu::Array* newArrays = size == 0 ? 0 : new cu::Array[size];

#if 1
    // Move the old arrays into the new ones
    for (size_t n = 0; n < numVolumeArrays && n < size; ++n)
    {
      newArrays[n].reset(volumeArrays[n].release());
    }
#endif

    // Delete the old arrays
    delete [] volumeArrays;

    // Update members
    volumeArrays = newArrays;
    numVolumeArrays = size;
  }
};


vvRayRend::vvRayRend(vvVolDesc* vd, vvRenderState renderState)
  : BaseType(vd, renderState)
  , impl(new Impl)
{

  rendererType = RAYREND;

  glewInit();

  virvo::RenderTarget* rt = virvo::PixelUnpackBufferRT::create(virvo::PF_RGBA8, virvo::PF_LUMINANCE8);

  // no direct rendering
  if (rt == NULL)
  {
    rt = virvo::DeviceBufferRT::create(virvo::PF_RGBA8, virvo::PF_LUMINANCE8);
  }
  setRenderTarget(rt);

  bool ok;

  // Free "cuda error cache".
  virvo::cuda::checkError(&ok, cudaGetLastError(), "vvRayRend::vvRayRend() - free cuda error cache");

  _volumeCopyToGpuOk = true;

  _twoPassIbr = (_ibrMode == VV_REL_THRESHOLD || _ibrMode == VV_EN_EX_MEAN);

  _rgbaTF = NULL;

  initVolumeTexture();

  updateTransferFunction();
}

vvRayRend::~vvRayRend()
{
  vvDebugMsg::msg(1, "vvRayRend::~vvRayRend()");

  delete[] _rgbaTF;
}

size_t vvRayRend::getLUTSize() const
{
   vvDebugMsg::msg(2, "vvRayRend::getLUTSize()");
   return (vd->getBPV()==2) ? 4096 : 256;
}

bool vvRayRend::instantClassification() const
{
  return true;
}


void vvRayRend::setVolDesc(vvVolDesc* vd)
{
  vvRenderer::setVolDesc(vd);
  initVolumeTexture();
}


void vvRayRend::updateTransferFunction()
{
  vvDebugMsg::msg(3, "vvRayRend::updateTransferFunction()");

  size_t lutEntries = getLUTSize();
  delete[] _rgbaTF;
  _rgbaTF = new float[4 * lutEntries];

  vd->computeTFTexture(lutEntries, 1, 1, _rgbaTF);

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

  if (!impl->transferFuncArray.allocate(channelDesc, lutEntries, 1))
  {
  }
  if (!impl->transferFuncArray.upload(0, 0, _rgbaTF, lutEntries * 4 * sizeof(float)))
  {
  }

  cTFTexture.setFilterMode(cudaFilterModeLinear);
  cTFTexture.setNormalized(true);    // access with normalized texture coordinates
  cTFTexture.setAddressMode(cudaAddressModeClamp);   // wrap texture coordinates

  cTFTexture.bind(impl->transferFuncArray, channelDesc);
}

void vvRayRend::renderVolumeGL()
{
  vvDebugMsg::msg(3, "vvRayRend::compositeVolume()");

  if(!_volumeCopyToGpuOk)
  {
    std::cerr << "vvRayRend::compositeVolume() aborted because of previous CUDA-Error" << std::endl;
    return;
  }

  vec3f size = vd->getSize();

  vec3f probePosObj;
  vec3f probeSizeObj;
  vec3f probeMin;
  vec3f probeMax;
  calcProbeDims(probePosObj, probeSizeObj, probeMin, probeMax);

  vec3f clippedProbeSizeObj = probeSizeObj;
  for (int i=0; i<3; ++i)
  {
    if (clippedProbeSizeObj[i] < vd->getSize()[i])
    {
      clippedProbeSizeObj[i] = vd->getSize()[i];
    }
  }

  if (_isROIUsed && !_sphericalROI)
  {
    drawBoundingBox(probeSizeObj, roi_pos_, _probeColor);
  }

  const float diagonalVoxels = sqrtf(float(vd->vox[0] * vd->vox[0] +
                                           vd->vox[1] * vd->vox[1] +
                                           vd->vox[2] * vd->vox[2]));
  size_t numSlices = std::max(size_t(1), static_cast<size_t>(_quality * diagonalVoxels));

  mat4 mv;
  mat4 pr;

#ifdef HAVE_OPENGL
  mv = virvo::gl::getModelviewMatrix();
  pr = virvo::gl::getProjectionMatrix();
#endif

  mat4 inv_view_matrix = inverse( pr * mv );
  cInvViewMatrix.update(inv_view_matrix.data(), 16 * sizeof(float));

  mat4 MvPr = pr * mv;
  cMvPrMatrix.update(MvPr.data(), 16 * sizeof(float));

  const float3 volPos = make_float3(vd->pos[0], vd->pos[1], vd->pos[2]);
  float3 probePos = volPos;
  if (_isROIUsed && !_sphericalROI)
  {
    probePos = make_float3(probePosObj[0],  probePosObj[1], probePosObj[2]);
  }
  const float3 volSize = make_float3(size[0], size[1], size[2]);
  float3 probeSize = make_float3(probeSizeObj[0], probeSizeObj[1], probeSizeObj[2]);
  if (_sphericalROI)
  {
    probeSize = make_float3((float)vd->vox[0], (float)vd->vox[1], (float)vd->vox[2]);
  }

  float3 texsize = make_float3(vd->vox[0], vd->vox[1], vd->vox[2]);

  const bool isOrtho = pr(3, 0) == 0.0f && pr(3, 1) == 0.0f && pr(3, 2) == 0.0f;

  vec3f eye = getEyePosition();

    gl::light l;
    if (glIsEnabled(GL_LIGHTING))
    {
        l = gl::getLight(GL_LIGHT0);
    }

  // position of light source. transform eye coordinates to object coordinates
  vec4 LposEye = inverse(mv) * l.position;
  const float3 Lpos = make_float3(LposEye.x, LposEye.y, LposEye.z);

  vec3f normal;
  vec3f origin;
  getShadingNormal(normal, origin, eye, inverse(mv), isOrtho);

  // viewing direction equals normal direction
  const float3 V = make_float3(normal[0], normal[1], normal[2]);

  // Clip sphere.
  const float3 center = make_float3(clip_sphere_center_[0], clip_sphere_center_[1], clip_sphere_center_[2]);
  const float radius  = _clipSphereRadius;

  // Clip plane.
  const float3 pnormal = normalize( make_float3(clip_plane_normal_[0], clip_plane_normal_[1], clip_plane_normal_[2]) );
  const float pdist = dot( -clip_plane_normal_, clip_plane_point_ );

  if (_clipMode == 1 && _clipPlanePerimeter)
  {
    drawPlanePerimeter(size, vd->pos, clip_plane_point_, clip_plane_normal_, _clipPlaneColor);
  }

  GLfloat bgcolor[4];
  glGetFloatv(GL_COLOR_CLEAR_VALUE, bgcolor);

  if (_interpolation == virvo::BSplineInterpol)
  {
    cPrefilterTexture.bind(impl->getVolumeArray(vd->getCurrentFrame()), impl->channelDesc);
  }
  else if (vd->bpc == 1)
  {
    cVolTexture8.bind(impl->getVolumeArray(vd->getCurrentFrame()), impl->channelDesc);
  }
  else if (vd->bpc == 2)
  {
    cVolTexture16.bind(impl->getVolumeArray(vd->getCurrentFrame()), impl->channelDesc);
  }

  RayRendKernelParams kernelParams;

  kernelParams.blockDimX            = 8;
  kernelParams.blockDimY            = 8;
  kernelParams.bpc                  = getVolDesc()->bpc;
  kernelParams.illumination         = _lighting;
  kernelParams.opacityCorrection    = _opacityCorrection;
  kernelParams.earlyRayTermination  = _earlyRayTermination;
  kernelParams.clipMode             = getParameter(vvRenderState::VV_CLIP_MODE);
  kernelParams.mipMode              = getParameter(vvRenderState::VV_MIP_MODE);
  kernelParams.useIbr               = getParameter(vvRenderState::VV_USE_IBR);

  virvo::RenderTarget* rt = getRenderTarget();

  assert(rt);

  void* deviceColor = rt->deviceColor();
  void* deviceDepth = rt->deviceDepth();

  assert(deviceColor);

  CallRayRendKernel(kernelParams,
                    reinterpret_cast<uchar4*>(deviceColor),
                    rt->width(),
                    rt->height(),
                    rt->width(),
                    diagonalVoxels / (float)numSlices,
                    volPos,
                    volSize * 0.5f,
                    probePos,
                    probeSize * 0.5f,
                    texsize,
                    _interpolation,
                    Lpos,
                    V,
                    l.constant_attenuation,
                    l.linear_attenuation,
                    l.quadratic_attenuation,
                    kernelParams.clipMode == 1,
                    kernelParams.clipMode == 2,
                    false,
                    center,
                    radius * radius,
                    pnormal,
                    pdist,
                    deviceDepth,
                    8 * getPixelSize(rt->depthFormat()),
                    make_float2(depth_range_[0], depth_range_[1]),
                    getIbrMode(_ibrMode),
                    _twoPassIbr
                    );
}


bool vvRayRend::checkParameter(ParameterType param, vvParam const& value) const
{
  switch (param)
  {
  case VV_SLICEINT:

    {
      virvo::tex_filter_mode mode = static_cast< virvo::tex_filter_mode >(value.asInt());

      if (mode == virvo::Nearest || mode == virvo::Linear
#if DESKVOX_USE_RAYREND_BSPLINE
       || mode == virvo::BSpline || mode == virvo::BSplineInterpol
#endif
#if DESKVOX_USE_RAYREND_CSPLINE
       || mode == virvo::CardinalSpline
#endif
       )
      {
        return true;
      }
    }

    return false;

  default:

    return vvRenderer::checkParameter(param, value);

  }
}


//----------------------------------------------------------------------------
// see parent
void vvRayRend::setParameter(ParameterType param, const vvParam& newValue)
{
  vvDebugMsg::msg(3, "vvRayRend::setParameter()");

  switch (param)
  {
  case vvRenderer::VV_SLICEINT:
    {
      if (_interpolation != static_cast< virvo::tex_filter_mode >(newValue.asInt()))
      {
        _interpolation = static_cast< virvo::tex_filter_mode >(newValue.asInt());
        initVolumeTexture();
        updateTransferFunction();
      }
    }
    break;
  default:
    BaseType::setParameter(param, newValue);
    break;
  }
}

//----------------------------------------------------------------------------
// see parent
vvParam vvRayRend::getParameter(ParameterType param) const
{
  vvDebugMsg::msg(3, "vvRayRend::getParameter()");

  switch (param)
  {
  case vvRenderer::VV_SLICEINT:
    return _interpolation;
  default:
    return BaseType::getParameter(param);
  }
}

void vvRayRend::initVolumeTexture()
{
  vvDebugMsg::msg(3, "vvRayRend::initVolumeTexture()");

  bool ok;

  cudaExtent volumeSize = make_cudaExtent(vd->vox[0], vd->vox[1], vd->vox[2]);
  if (_interpolation == virvo::BSplineInterpol)
  {
    impl->channelDesc = cudaCreateChannelDesc<short>();
  }
  else if (vd->bpc == 1)
  {
    impl->channelDesc = cudaCreateChannelDesc<uchar>();
  }
  else if (vd->bpc == 2)
  {
    impl->channelDesc = cudaCreateChannelDesc<ushort>();
  }

  impl->allocateVolumeArrays(vd->frames);

  bool outOfMem = false;
  size_t outOfMemFrame = 0;
  for (size_t f=0; f<vd->frames; ++f)
  {
    _volumeCopyToGpuOk = impl->getVolumeArray(f).allocate3D(impl->channelDesc, volumeSize);

    size_t availableMem;
    size_t totalMem;
    virvo::cuda::checkError(&ok, cudaMemGetInfo(&availableMem, &totalMem),
                       "vvRayRend::initVolumeTexture() - get mem info from device");

    if(!_volumeCopyToGpuOk)
    {
      outOfMem = true;
      outOfMemFrame = f;
      break;
    }

    VV_LOG(1) << "Total CUDA memory (MB):     " << (size_t)(totalMem/1024/1024) << std::endl;
    VV_LOG(1) << "Available CUDA memory (MB): " << (size_t)(availableMem/1024/1024) << std::endl;

    cudaMemcpy3DParms copyParams = { 0 };

    virvo::texture< uint8_t, virvo::ElementType, 3 > tex( vd->vox[0], vd->vox[1], vd->vox[2] );
    if (_interpolation == virvo::BSplineInterpol)
    {
      tex.data = vd->getRaw(f);
      tex.set_filter_mode(virvo::BSplineInterpol);
      copyParams.srcPtr = make_cudaPitchedPtr(tex.prefiltered_data, volumeSize.width * sizeof(short), volumeSize.width, volumeSize.height);
    }
    else if (vd->bpc == 1 || vd->bpc == 2)
    {
      copyParams.srcPtr = make_cudaPitchedPtr(vd->getRaw(f), volumeSize.width*vd->bpc, volumeSize.width, volumeSize.height);
    }
    copyParams.dstArray = impl->getVolumeArray(f).get();
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyHostToDevice;
    virvo::cuda::checkError(&ok, cudaMemcpy3D(&copyParams),
                       "vvRayRend::initVolumeTexture() - copy volume frame to 3D array");
  }

  if (outOfMem)
  {
    std::cerr << "Could not accommodate the volume" << std::endl;
    for (size_t f=0; f<=outOfMemFrame; ++f)
    {
      impl->getVolumeArray(f).reset();
    }
  }

  if (vd->bpc == 1)
  {
    //
    // XXX:
    // why do we do this right here?
    //
    for (size_t f=0; f<outOfMemFrame; ++f)
    {
      impl->getVolumeArray(f).reset();
    }
  }

  if (_volumeCopyToGpuOk)
  {

    cudaTextureFilterMode filter_mode  = cudaFilterModePoint;

    switch (_interpolation)
    {
    case virvo::Nearest:

      filter_mode = cudaFilterModePoint;
      break;

    case virvo::Linear:

      filter_mode = cudaFilterModeLinear;
      break;

    case virvo::BSpline:

      filter_mode = cudaFilterModeLinear;
      break;

    case virvo::BSplineInterpol:

      filter_mode = cudaFilterModeLinear;
      break;

    case virvo::CardinalSpline:

      filter_mode = cudaFilterModePoint;
      break;

    }


    if (_interpolation == virvo::BSplineInterpol)
    {
        cPrefilterTexture.setNormalized(true);
        cPrefilterTexture.setFilterMode(filter_mode);
        cPrefilterTexture.setAddressMode(cudaAddressModeClamp);

        ok = cPrefilterTexture.bind(impl->getVolumeArray(0), impl->channelDesc);
    }
    else if (vd->bpc == 1)
    {
        cVolTexture8.setNormalized( _interpolation != virvo::CardinalSpline );
        cVolTexture8.setFilterMode(filter_mode);
        cVolTexture8.setAddressMode(cudaAddressModeClamp);

        ok = cVolTexture8.bind(impl->getVolumeArray(0), impl->channelDesc);
    }
    else if (vd->bpc == 2)
    {
        cVolTexture16.setNormalized( _interpolation != virvo::CardinalSpline );
        cVolTexture16.setFilterMode(filter_mode);
        cVolTexture16.setAddressMode(cudaAddressModeClamp);

        ok = cVolTexture16.bind(impl->getVolumeArray(0), impl->channelDesc);
    }
  }
}

vvRenderer* createRayRend(vvVolDesc* vd, vvRenderState const& rs)
{
  return new vvRayRend(vd, rs);
}

#endif // HAVE_CUDA
