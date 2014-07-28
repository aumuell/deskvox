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

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include <GL/glew.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <limits.h>
#include <math.h>
#include <cstring>

#include "vvopengl.h"
#include "vvdynlib.h"

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvvecmath.h"
#include "vvdebugmsg.h"
#include "vvtoolshed.h"
#include "vvtexrend.h"
#include "vvprintgl.h"
#include "vvshaderfactory.h"
#include "vvshaderprogram.h"
#include "vvvoldesc.h"
#include "vvpthread.h"

#include "gl/util.h"

#include "private/vvgltools.h"
#include "private/vvlog.h"

using namespace std;

namespace gl = virvo::gl;

using virvo::mat4;
using virvo::vec3f;
using virvo::vec3;
using virvo::vec4f;
using virvo::vec4i;
using virvo::vec4;

using virvo::PixelFormat;

enum {
  Shader1Chan = 0,
  ShaderPreInt = 11,
  ShaderLighting = 12,
  ShaderMultiTF = 13,
};


//----------------------------------------------------------------------------
const int vvTexRend::NUM_PIXEL_SHADERS = ShaderMultiTF+1;

static virvo::BufferPrecision mapBitsToBufferPrecision(int bits)
{
    switch (bits)
    {
    case 8:
        return virvo::Byte;
    case 16:
        return virvo::Short;
    case 32:
        return virvo::Float;
    default:
        assert(!"unknown bit size");
        return virvo::Byte;
    }
}

static PixelFormat mapBufferPrecisionToFormat(virvo::BufferPrecision bp)
{
    switch (bp)
    {
    case virvo::Byte:
        return virvo::PF_RGBA8;
    case virvo::Short:
        return virvo::PF_RGBA16F;
    case virvo::Float:
        return virvo::PF_RGBA32F;
    default:
        assert(!"unknown format");
        return virvo::PF_UNSPECIFIED;
    }
}

//----------------------------------------------------------------------------
/** Constructor.
  @param vd                      volume description
  @param renderState             object describing the render state
  @param geom                    render geometry (default: automatic)
  @param vox                     voxel type (default: best)
  @param displayNames            names of x-displays (host:display.screen) for multi-gpu rendering
  @param numDisplays             # displays for multi-gpu rendering
  @param multiGpuBufferPrecision precision of the offscreen buffer used for multi-gpu rendering
*/
vvTexRend::vvTexRend(vvVolDesc* vd, vvRenderState renderState, VoxelType vox)
  : vvRenderer(vd, renderState)
{
  vvDebugMsg::msg(1, "vvTexRend::vvTexRend()");

  glewInit();

  if (this->_useOffscreenBuffer)
    setRenderTarget( virvo::FramebufferObjectRT::create( mapBufferPrecisionToFormat(this->_imagePrecision), virvo::PF_DEPTH24_STENCIL8) );

  if (vvDebugMsg::isActive(2))
  {
#ifdef _WIN32
    cerr << "_WIN32 is defined" << endl;
#elif WIN32
    cerr << "_WIN32 is not defined, but should be if running under Windows" << endl;
#endif

#ifdef HAVE_CG
    cerr << "HAVE_CG is defined" << endl;
#else
    cerr << "Tip: define HAVE_CG for pixel shader support" << endl;
#endif

    cerr << "Compiler knows OpenGL versions: ";
#ifdef GL_VERSION_1_1
    cerr << "1.1";
#endif
#ifdef GL_VERSION_1_2
    cerr << ", 1.2";
#endif
#ifdef GL_VERSION_1_3
    cerr << ", 1.3";
#endif
#ifdef GL_VERSION_1_4
    cerr << ", 1.4";
#endif
#ifdef GL_VERSION_1_5
    cerr << ", 1.5";
#endif
#ifdef GL_VERSION_2_0
    cerr << ", 2.0";
#endif
#ifdef GL_VERSION_2_1
    cerr << ", 2.1";
#endif
#ifdef GL_VERSION_3_0
    cerr << ", 3.0";
#endif
#ifdef GL_VERSION_3_1
    cerr << ", 3.1";
#endif
#ifdef GL_VERSION_3_2
    cerr << ", 3.2";
#endif
#ifdef GL_VERSION_3_3
    cerr << ", 3.3";
#endif
#ifdef GL_VERSION_4_0
    cerr << ", 4.0";
#endif
#ifdef GL_VERSION_4_1
    cerr << ", 4.1";
#endif
#ifdef GL_VERSION_4_2
    cerr << ", 4.2";
#endif
    cerr << endl;
  }

  _shaderFactory = new vvShaderFactory();

  rendererType = TEXREND;
  texNames = NULL;
  _sliceOrientation = VV_VARIABLE;
  minSlice = maxSlice = -1;
  rgbaTF.resize(1);
  rgbaTF[0].resize(256 * 256 * 4);
  rgbaLUT.resize(1);
  rgbaLUT[0].resize(256 * 256 * 4);
  preintTable = new uint8_t[getPreintTableSize()*getPreintTableSize()*4];
  usePreIntegration = false;
  textures = 0;

  _currentShader = vd->chan - 1;
  _previousShader = _currentShader;

  if (vd->tf.size() > 1)
  {
    _currentShader = ShaderMultiTF; // TF for each channel
  }

  _lastFrame = std::numeric_limits<size_t>::max();
  lutDistance = -1.0;
  _isROIChanged = true;

  // Find out which OpenGL extensions are supported:
  extTex3d  = vvGLTools::isGLextensionSupported("GL_EXT_texture3D") || vvGLTools::isGLVersionSupported(1,2,1);
  arbMltTex = vvGLTools::isGLextensionSupported("GL_ARB_multitexture") || vvGLTools::isGLVersionSupported(1,3,0);

  extMinMax = vvGLTools::isGLextensionSupported("GL_EXT_blend_minmax") || vvGLTools::isGLVersionSupported(1,4,0);
  extBlendEquation = vvGLTools::isGLextensionSupported("GL_EXT_blend_equation") || vvGLTools::isGLVersionSupported(1,1,0);
  extPixShd  = isSupported(VV_PIX_SHD);

  extNonPower2 = vvGLTools::isGLextensionSupported("GL_ARB_texture_non_power_of_two") || vvGLTools::isGLVersionSupported(2,0,0);

  // Determine best rendering algorithm for current hardware:
  setVoxelType(findBestVoxelType(vox));

  _shader = initShader();
  if(voxelType == VV_PIX_SHD && !_shader)
    setVoxelType(VV_RGBA);
  initClassificationStage();

  cerr << "Rendering algorithm: ";
  switch(voxelType)
  {
    case VV_RGBA:    cerr << "VV_RGBA";    break;
    case VV_PIX_SHD: cerr << "VV_PIX_SHD, vv_shader" << _currentShader+1; break;
    default: assert(0); break;
  }
  cerr << endl;

  textures = 0;

  if (voxelType != VV_RGBA)
  {
    makeTextures(true);      // we only have to do this once for non-RGBA textures
  }
  updateTransferFunction();
}

//----------------------------------------------------------------------------
/// Destructor
vvTexRend::~vvTexRend()
{
  vvDebugMsg::msg(1, "vvTexRend::~vvTexRend()");

  freeClassificationStage();
  removeTextures();

  delete _shader;
  _shader = NULL;

  delete[] preintTable;
}


void vvTexRend::setVolDesc(vvVolDesc* vd)
{
  vvRenderer::setVolDesc(vd);
  makeTextures(true);
}


//------------------------------------------------
/** Initialize texture parameters for a voxel type
  @param vt voxeltype
*/
void vvTexRend::setVoxelType(vvTexRend::VoxelType vt)
{
  voxelType = vt;
  switch(voxelType)
  {
    case VV_PIX_SHD:
      if(vd->chan == 1)
      {
        texelsize=1;
        internalTexFormat = GL_LUMINANCE;
        texFormat = GL_LUMINANCE;
      }
      else if (vd->chan == 2)
      {
        texelsize=2;
        internalTexFormat = GL_LUMINANCE_ALPHA;
        texFormat = GL_LUMINANCE_ALPHA;
      }
      else if (vd->chan == 3)
      {
        texelsize=3;
        internalTexFormat = GL_RGB;
        texFormat = GL_RGB;
      }
      else
      {
        texelsize=4;
        internalTexFormat = GL_RGBA;
        texFormat = GL_RGBA;
      }
      break;
    case VV_RGBA:
      internalTexFormat = GL_RGBA;
      texFormat = GL_RGBA;
      texelsize=4;
      break;
    default:
      assert(0);
      break;
  }
}

//----------------------------------------------------------------------------
/// Chooses the best voxel type depending on the graphics hardware's
/// capabilities.
vvTexRend::VoxelType vvTexRend::findBestVoxelType(const vvTexRend::VoxelType vox) const
{
  vvDebugMsg::msg(1, "vvTexRend::findBestVoxelType()");

  if (vox==VV_BEST)
  {
    if (vd->chan==1)
    {
      if (extPixShd) return VV_PIX_SHD;
    }
    else
    {
      if (extPixShd) return VV_PIX_SHD;
    }
    return VV_RGBA;
  }
  else
  {
    switch(vox)
    {
      case VV_PIX_SHD: if (extPixShd) return VV_PIX_SHD;
      default: return VV_RGBA;
    }
  }
}

//----------------------------------------------------------------------------
/// Remove all textures from texture memory.
void vvTexRend::removeTextures()
{
  vvDebugMsg::msg(1, "vvTexRend::removeTextures()");

  if (textures > 0)
  {
    glDeleteTextures(textures, texNames);
    delete[] texNames;
    texNames = NULL;
    textures = 0;
  }
}

//----------------------------------------------------------------------------
/// Generate textures for all rendering modes.
vvTexRend::ErrorType vvTexRend::makeTextures(bool newTex)
{
  ErrorType err = OK;

  vvDebugMsg::msg(2, "vvTexRend::makeTextures()");

  vvssize3 vox = _paddingRegion.getMax() - _paddingRegion.getMin();
  for (size_t i = 0; i < 3; ++i)
  {
    vox[i] = std::min(vox[i], vd->vox[i]);
  }

  if (vox[0] == 0 || vox[1] == 0 || vox[2] == 0)
    return err;

  // Compute texture dimensions (perhaps must be power of 2):
  texels[0] = getTextureSize(vox[0]);
  texels[1] = getTextureSize(vox[1]);
  texels[2] = getTextureSize(vox[2]);

  updateTextures3D(0, 0, 0, texels[0], texels[1], texels[2], newTex);
  vvGLTools::printGLError("vvTexRend::makeTextures");

  if (voxelType==VV_PIX_SHD)
  {
    updateTransferFunction();
    updateLUT(1.f);
    makeLUTTexture();
  }
  return err;
}

//----------------------------------------------------------------------------
/// Generate texture for look-up table.
void vvTexRend::makeLUTTexture() const
{
  vvsize3 size;

  vvGLTools::printGLError("enter makeLUTTexture");
  for (size_t chan=0; chan<pixLUTName.size(); ++chan)
  {
    getLUTSize(size);
    glBindTexture(GL_TEXTURE_2D, pixLUTName[chan]);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size[0], size[1], 0,
        GL_RGBA, GL_UNSIGNED_BYTE, &rgbaLUT[chan][0]);
  }
  vvGLTools::printGLError("leave makeLUTTexture");
}

void vvTexRend::setTexMemorySize(size_t newSize)
{
  if (_texMemorySize == newSize)
    return;

  _texMemorySize = newSize;
}

size_t vvTexRend::getTexMemorySize() const
{
  return _texMemorySize;
}

//----------------------------------------------------------------------------
/// Update transfer function from volume description.
void vvTexRend::updateTransferFunction()
{
  vvsize3 size;

  vvDebugMsg::msg(1, "vvTexRend::updateTransferFunction()");
  if (voxelType==VV_PIX_SHD && vd->tf.size() > 1 )
  {
     usePreIntegration = false;
     _currentShader = ShaderMultiTF;
  }
  else
  {
     if (_preIntegration &&
           arbMltTex && 
           !(_clipMode == 1 && (_clipSingleSlice || _clipOpaque)) &&
           (voxelType==VV_PIX_SHD && (_currentShader==Shader1Chan || _currentShader==ShaderPreInt)))
     {
        usePreIntegration = true;
        if(_currentShader==Shader1Chan)
           _currentShader = ShaderPreInt;
     }
     else
     {
        usePreIntegration = false;
        if(_currentShader==ShaderPreInt)
           _currentShader = Shader1Chan;
     }
  }

  // Generate arrays from pins:
  getLUTSize(size);
  if (vd->tf.size() != rgbaTF.size())
    rgbaTF.resize(vd->tf.size());
  for (size_t i=0; i<vd->tf.size(); ++i) {
    if (rgbaTF[i].size() != size[0]*size[1]*size[2]*4) // reserve space for TF as 4 floats/entry (RGBA)
       rgbaTF[i].resize(size[0]*size[1]*size[2]*4);
    vd->computeTFTexture(i, size[0], size[1], size[2], &rgbaTF[i][0]);
  }

  if(!instantClassification())
    updateLUT(1.0f);                                // generate color/alpha lookup table
  else
    lutDistance = -1.;                              // invalidate LUT
}

//----------------------------------------------------------------------------
// see parent in vvRenderer
void vvTexRend::updateVolumeData()
{
  vvRenderer::updateVolumeData();

  makeTextures(true);
}

//----------------------------------------------------------------------------
void vvTexRend::updateVolumeData(size_t offsetX, size_t offsetY, size_t offsetZ,
                                 size_t sizeX, size_t sizeY, size_t sizeZ)
{
  updateTextures3D(offsetX, offsetY, offsetZ, sizeX, sizeY, sizeZ, false);
}

//----------------------------------------------------------------------------
/**
   Method to create a new 3D texture or update parts of an existing 3D texture.
   @param offsetX, offsetY, offsetZ: lower left corner of texture
   @param sizeX, sizeY, sizeZ: size of texture
   @param newTex: true: create a new texture
                  false: update an existing texture
*/
vvTexRend::ErrorType vvTexRend::updateTextures3D(ssize_t offsetX, ssize_t offsetY, ssize_t offsetZ,
                                                 ssize_t sizeX, ssize_t sizeY, ssize_t sizeZ, bool newTex)
{
  ErrorType err = OK;
  size_t srcIndex;
  size_t texOffset=0;
  vec4i rawVal;
  uint8_t* texData = NULL;
  bool accommodated = true;
  GLint glWidth;

  vvDebugMsg::msg(1, "vvTexRend::updateTextures3D()");

  if (!extTex3d) return NO3DTEX;

  size_t texSize = sizeX * sizeY * sizeZ * texelsize;
  VV_LOG(1) << "3D Texture width     = " << sizeX << std::endl;
  VV_LOG(1) << "3D Texture height    = " << sizeY << std::endl;
  VV_LOG(1) << "3D Texture depth     = " << sizeZ << std::endl;
  VV_LOG(1) << "3D Texture size (KB) = " << texSize / 1024 << std::endl;

  size_t sliceSize = vd->getSliceBytes();

  if (vd->frames != textures)
    newTex = true;

  if (newTex)
  {
    VV_LOG(2) << "Creating texture names. # of names: " << vd->frames << std::endl;

    removeTextures();
    textures  = vd->frames;
    delete[] texNames;
    texNames = new GLuint[textures];
    glGenTextures(vd->frames, texNames);
  }

  VV_LOG(2) << "Transferring textures to TRAM. Total size [KB]: " << vd->frames * texSize / 1024 << std::endl;

  vvssize3 offsets(offsetX, offsetY, offsetZ);
  offsets += _paddingRegion.getMin();

  bool useRaw = vd->bpc==1 && vd->chan<=4 && vd->chan==texelsize;
  if (sizeX != vd->vox[0])
    useRaw = false;
  if (sizeY != vd->vox[1])
    useRaw = false;
  if (sizeZ != vd->vox[2])
    useRaw = false;
  for (int i=0; i<3; ++i) {
    if (offsets[i] != 0)
      useRaw = false;
  }
  if (!useRaw)
  {
    texData = new uint8_t[texSize];
    memset(texData, 0, texSize);
  }

  // Generate sub texture contents:
  for (size_t f = 0; f < vd->frames; f++)
  {
    uint8_t *raw = vd->getRaw(f);
    if (useRaw) {
      texData = raw;
    }
    else
    {
      for (ssize_t s = offsets[2]; s < (offsets[2] + sizeZ); s++)
      {
        size_t rawSliceOffset = (ts_min(ts_max(s,ssize_t(0)),vd->vox[2]-1)) * sliceSize;
        for (ssize_t y = offsets[1]; y < (offsets[1] + sizeY); y++)
        {
          size_t heightOffset = (ts_min(ts_max(y,ssize_t(0)),vd->vox[1]-1)) * vd->vox[0] * vd->bpc * vd->chan;
          size_t texLineOffset = (y - offsets[1] - offsetY) * sizeX + (s - offsets[2] - offsetZ) * sizeX * sizeY;
          
          if (vd->chan == 1 && (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4))
          {
            if (vd->bpc == 1 && texelsize == 1)
            {
              // one byte, one color channel ==> can use memcpy for consecutive memory chunks
              ssize_t x1 = offsets[0];
              ssize_t x2 = offsets[0] + sizeX;
              size_t srcMin = vd->bpc * min(x1, vd->vox[0] - 1) + rawSliceOffset + heightOffset;
              size_t srcMax = vd->bpc * min(x2, vd->vox[0] - 1) + rawSliceOffset + heightOffset;
              texOffset = texLineOffset - offsetX;
              memcpy(&texData[texelsize * texOffset], &raw[srcMin], srcMax - srcMin);
            }
            else
            {
              for (ssize_t x = offsets[0]; x < (offsets[0] + sizeX); x++)
              {
                srcIndex = vd->bpc * min(x,vd->vox[0]-1) + rawSliceOffset + heightOffset;
                if (vd->bpc == 1) rawVal[0] = int(raw[srcIndex]);
                else if (vd->bpc == 2)
                {
                  rawVal[0] = *(uint16_t*)(raw+srcIndex);
                  rawVal[0] >>= 8;
                }
                else // vd->bpc==4: convert floating point to 8bit value
                {
                  const float fval = *((float*)(raw + srcIndex));      // fetch floating point data value
                  rawVal[0] = vd->mapFloat2Int(fval);
                }
                texOffset = (x - offsets[0] - offsetX) + texLineOffset;
                switch(voxelType)
                {
                case VV_PIX_SHD:
                  texData[texelsize * texOffset] = (uint8_t) rawVal[0];
                  break;
                case VV_RGBA:
                  for (size_t c = 0; c < 4; c++)
                  {
                    texData[4 * texOffset + c] = rgbaLUT[0][size_t(rawVal[0]) * 4 + c];
                  }
                  break;
                default:
                  assert(0);
                  break;
                }
              }
            }
          }
          else if (vd->bpc==1 || vd->bpc==2 || vd->bpc==4)
          {
            if (voxelType == VV_RGBA || voxelType == VV_PIX_SHD)
            {
              for (ssize_t x = offsets[0]; x < (offsets[0] + sizeX); x++)
              {
                texOffset = (x - offsets[0] - offsetX) + texLineOffset;
                for (size_t c = 0; c < ts_min(vd->chan, size_t(4)); c++)
                {
                  srcIndex = vd->bpc * (min(x,vd->vox[0]-1)*vd->chan+c) + rawSliceOffset + heightOffset;
                  if (vd->bpc == 1)
                    rawVal[c] = (int) raw[srcIndex];
                  else if (vd->bpc == 2)
                  {
                    rawVal[c] = *((uint16_t *)(raw + srcIndex));
                    rawVal[c] >>= 8;
                  }
                  else  // vd->bpc == 4
                  {
                    const float fval = *((float*)(raw + srcIndex));      // fetch floating point data value
                    rawVal[c] = vd->mapFloat2Int(fval);
                  }
                }

                // Copy color components:
                for (size_t c = 0; c < ts_min(vd->chan, size_t(3)); c++)
                {
                  texData[4 * texOffset + c] = (uint8_t) rawVal[c];
                }
              }

              // Alpha channel:
              if (vd->chan >= 4)
              {
                texData[4 * texOffset + 3] = (uint8_t)rawVal[3];
              }
              else
              {
                size_t alpha = 0;
                for (size_t c = 0; c < vd->chan; c++)
                {
                  // Alpha: mean of sum of RGB conversion table results:
                  alpha += (size_t) rgbaLUT[0][size_t(rawVal[c]) * 4 + c];
                }
                texData[4 * texOffset + 3] = (uint8_t) (alpha / vd->chan);
              }
            }
          }
          else cerr << "Cannot create texture: unsupported voxel format (3)." << endl;
        }
      }
    }

    if (newTex)
    {
      glBindTexture(GL_TEXTURE_3D_EXT, texNames[f]);
      glPixelStorei(GL_UNPACK_ALIGNMENT,1);
      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT, GL_CLAMP);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP);

      glTexImage3D(GL_PROXY_TEXTURE_3D_EXT, 0, internalTexFormat,
        texels[0], texels[1], texels[2], 0, texFormat, GL_UNSIGNED_BYTE, NULL);
      glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D_EXT, 0, GL_TEXTURE_WIDTH, &glWidth);

      if (glWidth==texels[0])
      {
        glTexImage3D(GL_TEXTURE_3D_EXT, 0, internalTexFormat, texels[0], texels[1], texels[2], 0,
          texFormat, GL_UNSIGNED_BYTE, texData);
      }
      else
      {
        accommodated = false;
        vvGLTools::printGLError("Tried to accomodate 3D textures");

        cerr << "Insufficient texture memory for 3D texture(s)." << endl;
        err = TRAM_ERROR;
      }
    }
    else
    {
      glBindTexture(GL_TEXTURE_3D_EXT, texNames[f]);
      glTexSubImage3D(GL_TEXTURE_3D_EXT, 0, offsetX, offsetY, offsetZ,
        sizeX, sizeY, sizeZ, texFormat, GL_UNSIGNED_BYTE, texData);
    }
  }

  if (!useRaw)
  {
    delete[] texData;
  }
  return err;
}

//----------------------------------------------------------------------------
/// Set GL environment for texture rendering.
void vvTexRend::setGLenvironment() const
{
  vvDebugMsg::msg(3, "vvTexRend::setGLenvironment()");

  // Save current GL state:
  glPushAttrib(GL_COLOR_BUFFER_BIT | GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT
               | GL_ENABLE_BIT | GL_TEXTURE_BIT | GL_TRANSFORM_BIT);

  // Set new GL state:
  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);                           // default depth function
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_BLEND);

  if (glBlendFuncSeparate)
  {
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  }
  else
  {
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  }

  glMatrixMode(GL_TEXTURE);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glDepthMask(GL_FALSE);

  switch (_mipMode)
  {
    // alpha compositing
    case 0: glBlendEquation(GL_FUNC_ADD); break;
    case 1: glBlendEquation(GL_MAX); break;   // maximum intensity projection
    case 2: glBlendEquation(GL_MIN); break;   // minimum intensity projection
    default: break;
  }
  vvDebugMsg::msg(3, "vvTexRend::setGLenvironment() done");
}

//----------------------------------------------------------------------------
/// Unset GL environment for texture rendering.
void vvTexRend::unsetGLenvironment() const
{
  vvDebugMsg::msg(3, "vvTexRend::unsetGLenvironment()");

  glPopAttrib();

  vvDebugMsg::msg(3, "vvTexRend::unsetGLenvironment() done");
}

//----------------------------------------------------------------------------
/** Render a volume entirely if probeSize=0 or a cubic sub-volume of size probeSize.
  @param mv        model-view matrix
*/
void vvTexRend::renderTex3DPlanar(mat4 const& mv)
{
  vec3f vissize, vissize2;                        // full and half object visible sizes
  vvVector3 isect[6];                             // intersection points, maximum of 6 allowed when intersecting a plane and a volume [object space]
  vec3f texcoord[12];                             // intersection points in texture coordinate space [0..1]
  vec3f farthest;                                 // volume vertex farthest from the viewer
  vec3f delta;                                    // distance vector between textures [object space]
  vec3f normal;                                   // normal vector of textures
  vec3f origin;                                   // origin (0|0|0) transformed to object space
  vvVector3 normClipPoint;                        // normalized point on clipping plane
  vec3f clipPosObj;                               // clipping plane position in object space w/o position
  vec3f probePosObj;                              // probe midpoint [object space]
  vec3f probeSizeObj;                             // probe size [object space]
  vec3f probeTexels;                              // number of texels in each probe dimension
  vec3f probeMin, probeMax;                       // probe min and max coordinates [object space]
  vec3f texSize;                                  // size of 3D texture [object space]
  float     maxDist;                              // maximum length of texture drawing path
  size_t    numSlices;

  vvDebugMsg::msg(3, "vvTexRend::renderTex3DPlanar()");

  if (!extTex3d) return;                          // needs 3D texturing extension

  // determine visible size and half object size as shortcut
  vvssize3 minVox = _visibleRegion.getMin();
  vvssize3 maxVox = _visibleRegion.getMax();
  for (size_t i = 0; i < 3; ++i)
  {
    minVox[i] = std::max(minVox[i], ssize_t(0));
    maxVox[i] = std::min(maxVox[i], vd->vox[i]);
  }
  const vvVector3 minCorner = vd->objectCoords(minVox);
  const vvVector3 maxCorner = vd->objectCoords(maxVox);
  vissize = maxCorner - minCorner;
  vec3f center = vvAABB(minCorner, maxCorner).getCenter();

  for (size_t i=0; i<3; ++i)
  {
    texSize[i] = vissize[i] * (float)texels[i] / (float)vd->vox[i];
    vissize2[i]   = 0.5f * vissize[i];
  }
  vec3f pos = vd->pos + center;

  if (_isROIUsed)
  {
    vec3f size = vd->getSize();
    vec3f size2 = size * 0.5f;
    // Convert probe midpoint coordinates to object space w/o position:
    probePosObj = roi_pos_;
    probePosObj -= pos;                        // eliminate object position from probe position

    // Compute probe min/max coordinates in object space:
    probeMin = probePosObj - (roi_size_ * size) * 0.5f;
    probeMax = probePosObj + (roi_size_ * size) * 0.5f;

    // Constrain probe boundaries to volume data area:
    for (size_t i=0; i<3; ++i)
    {
      if (probeMin[i] > size2[i] || probeMax[i] < -size2[i])
      {
        vvDebugMsg::msg(3, "probe outside of volume");
        return;
      }
      if (probeMin[i] < -size2[i]) probeMin[i] = -size2[i];
      if (probeMax[i] >  size2[i]) probeMax[i] =  size2[i];
      probePosObj[i] = (probeMax[i] + probeMin[i]) *0.5f;
    }

    // Compute probe edge lengths:
    for (size_t i=0; i<3; ++i)
      probeSizeObj[i] = probeMax[i] - probeMin[i];
  }
  else                                            // probe mode off
  {
    probeSizeObj = vd->getSize();
    probeMin = minCorner;
    probeMax = maxCorner;
    probePosObj = center;
  }

  // Initialize texture counters
  if (_isROIUsed)
  {
    probeTexels = vec3f(0.0f, 0.0f, 0.0f);
    for (size_t i=0; i<3; ++i)
    {
      probeTexels[i] = texels[i] * probeSizeObj[i] / texSize[i];
    }
  }
  else                                            // probe mode off
  {
    probeTexels = vec3f( (float)vd->vox[0], (float)vd->vox[1], (float)vd->vox[2] );
  }

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glTranslatef(vd->pos[0], vd->pos[1], vd->pos[2]);

  // Calculate inverted modelview matrix:
  mat4 invMV = inverse(mv);

  // Find eye position (object space):
  vec3f eye = getEyePosition();

  // Get projection matrix:
  vvMatrix pm = gl::getProjectionMatrix();
  bool isOrtho = pm.isProjOrtho();

  getObjNormal(normal, origin, eye, invMV, isOrtho);
  evaluateLocalIllumination(_shader, normal);

  // compute number of slices to draw
  float depth = fabs(normal[0]*probeSizeObj[0]) + fabs(normal[1]*probeSizeObj[1]) + fabs(normal[2]*probeSizeObj[2]);
  size_t minDistanceInd = 0;
  if(probeSizeObj[1]/probeTexels[1] < probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd])
    minDistanceInd=1;
  if(probeSizeObj[2]/probeTexels[2] < probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd])
    minDistanceInd=2;
  float voxelDistance = probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd];

  float sliceDistance = voxelDistance / _quality;
  if(_isROIUsed && _quality < 2.0)
  {
    // draw at least twice as many slices as there are samples in the probe depth.
    sliceDistance = voxelDistance * 0.5f;
  }
  numSlices = 2*(size_t)ceilf(depth/sliceDistance*.5f);

  if (numSlices < 1)                              // make sure that at least one slice is drawn
    numSlices = 1;
  // don't render an insane amount of slices
  {
    vvssize3 sz = maxVox - minVox;
    ssize_t maxV = ts_max(sz[0], sz[1]);
    maxV = ts_max(maxV, sz[2]);
    ssize_t lim = maxV * 10. * ts_max(_quality, 1.f);
    if (numSlices > lim)
    {
      numSlices = lim;
      VV_LOG(1) << "Limiting number of slices to " << numSlices << std::endl;
    }
  }

  VV_LOG(3) << "Number of textures to render: " << numSlices << std::endl;

  // Use alpha correction in indexed mode: adapt alpha values to number of textures:
  if (instantClassification())
  {
    float thickness = sliceDistance/voxelDistance;

    // just tolerate slice distance differences imposed on us
    // by trying to keep the number of slices constant
    if(lutDistance/thickness < 0.88 || thickness/lutDistance < 0.88)
    {
      updateLUT(thickness);
    }
  }

  delta = normal;
  delta *= vec3f(sliceDistance);

  // Compute farthest point to draw texture at:
  farthest = delta;
  farthest *= vec3f((float)(numSlices - 1) * -0.5f);
  farthest += probePosObj; // will be vd->pos if no probe present

  if (_clipMode == 1)                     // clipping plane present?
  {
    // Adjust numSlices and set farthest point so that textures are only
    // drawn up to the clipPoint. (Delta may not be changed
    // due to the automatic opacity correction.)
    // First find point on clipping plane which is on a line perpendicular
    // to clipping plane and which traverses the origin:
    vec3f temp = delta * vec3f(-0.5f);
    farthest += temp;                          // add a half delta to farthest
    clipPosObj = clip_plane_point_;
    clipPosObj -= pos;
    temp = probePosObj;
    temp += normal;
    normClipPoint.isectPlaneLine(normal, clipPosObj, probePosObj, temp);
    maxDist = length( farthest - vec3f(normClipPoint) );
    numSlices = (size_t)( maxDist / length(delta) ) + 1;
    temp = delta;
    temp *= vec3f( ((float)(1 - static_cast<ptrdiff_t>(numSlices))) );
    farthest = normClipPoint;
    farthest += temp;
    if (_clipSingleSlice)
    {
      // Compute slice position:
      temp = delta;
      temp *= vec3f( ((float)(numSlices-1)) );
      farthest += temp;
      numSlices = 1;

      // Make slice opaque if possible:
      if (instantClassification())
      {
        updateLUT(0.0f);
      }
    }
  }

  vec3 texPoint;                                  // arbitrary point on current texture
  int drawn = 0;                                  // counter for drawn textures
  vec3 deltahalf = delta * 0.5f;

  // Relative viewing position
  vec3 releye = eye - pos;

  // Volume render a 3D texture:
  if(voxelType == VV_PIX_SHD && _shader)
  {
    enableShader(_shader);
    _shader->setParameterTex3D("pix3dtex", texNames[vd->getCurrentFrame()]);
  }
  else
  {
    enableTexture(GL_TEXTURE_3D_EXT);
    glBindTexture(GL_TEXTURE_3D_EXT, texNames[vd->getCurrentFrame()]);
  }

  texPoint = farthest;
  for (size_t i=0; i<numSlices; ++i)                     // loop thru all drawn textures
  {
    // Search for intersections between texture plane (defined by texPoint and
    // normal) and texture object (0..1):
    size_t isectCnt = isect->isectPlaneCuboid(normal, texPoint, probeMin, probeMax);

    texPoint += delta;

    if (isectCnt<3) continue;                     // at least 3 intersections needed for drawing

    // Check volume section mode:
    if (minSlice != -1 && static_cast<ptrdiff_t>(i) < minSlice) continue;
    if (maxSlice != -1 && static_cast<ptrdiff_t>(i) > maxSlice) continue;

    // Put the intersecting 3 to 6 vertices in cyclic order to draw adjacent
    // and non-overlapping triangles:
    isect->cyclicSort(isectCnt, normal);

    // Generate vertices in texture coordinates:
    if(usePreIntegration)
    {
      for (size_t j=0; j<isectCnt; ++j)
      {
        vvVector3 front, back;

        if(isOrtho)
        {
          back = isect[j];
          back.sub(deltahalf);
        }
        else
        {
          vvVector3 v = isect[j];
          v.sub(deltahalf);
          back.isectPlaneLine(normal, v, releye, isect[j]);
        }

        if(isOrtho)
        {
          front = isect[j];
          front.add(deltahalf);
        }
        else
        {
          vvVector3 v;
          v = isect[j];
          v.add(deltahalf);
          front.isectPlaneLine(normal, v, releye, isect[j]);
        }

        for (size_t k=0; k<3; ++k)
        {
          texcoord[j][k] = (back[k] - minCorner[k]) / vissize[k];
          texcoord[j][k] = texcoord[j][k] * (texMax[k] - texMin[k]) + texMin[k];

          texcoord[j+6][k] = (front[k] - minCorner[k]) / vissize[k];
          texcoord[j+6][k] = texcoord[j+6][k] * (texMax[k] - texMin[k]) + texMin[k];
        }
      }
    }
    else
    {
      for (size_t j=0; j<isectCnt; ++j)
      {
        for (size_t k=0; k<3; ++k)
        {
          texcoord[j][k] = (isect[j][k] - minCorner[k]) / vissize[k];
          texcoord[j][k] = texcoord[j][k] * (texMax[k] - texMin[k]) + texMin[k];
        }
      }
    }

    glBegin(GL_TRIANGLE_FAN);
    glColor4f(1.0, 1.0, 1.0, 1.0);
    glNormal3f(normal[0], normal[1], normal[2]);
    ++drawn;
    for (size_t j=0; j<isectCnt; ++j)
    {
      // The following lines are the bottleneck of this method:
      if(usePreIntegration)
      {
        glMultiTexCoord3fARB(GL_TEXTURE0_ARB, texcoord[j][0], texcoord[j][1], texcoord[j][2]);
        glMultiTexCoord3fARB(GL_TEXTURE1_ARB, texcoord[j+6][0], texcoord[j+6][1], texcoord[j+6][2]);
      }
      else
      {
        glTexCoord3f(texcoord[j][0], texcoord[j][1], texcoord[j][2]);
      }

      glVertex3f(isect[j][0], isect[j][1], isect[j][2]);
    }
    glEnd();
  }
  vvDebugMsg::msg(3, "Number of textures drawn: ", drawn);

  if (voxelType == VV_PIX_SHD && _shader)
  {
    disableShader(_shader);
  }
  else
  {
    disableTexture(GL_TEXTURE_3D_EXT);
  }
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

//----------------------------------------------------------------------------
/** Render the volume onto currently selected drawBuffer.
 Viewport size in world coordinates is -1.0 .. +1.0 in both x and y direction
*/
void vvTexRend::renderVolumeGL()
{
  vvDebugMsg::msg(3, "vvTexRend::renderVolumeGL()");

  vvGLTools::printGLError("enter vvTexRend::renderVolumeGL()");

  vvssize3 vox = _paddingRegion.getMax() - _paddingRegion.getMin();
  for (size_t i = 0; i < 3; ++i)
  {
    vox[i] = std::min(vox[i], vd->vox[i]);
  }

  if (vox[0] * vox[1] * vox[2] == 0)
    return;

  vec3f size = vd->getSize();                  // volume size [world coordinates]

  // Draw boundary lines (must be done before setGLenvironment()):
  if (_isROIUsed)
  {
    vec3f probeSizeObj = size * roi_size_;
    drawBoundingBox(probeSizeObj, roi_pos_, _probeColor);
  }
  if (_clipMode == 1 && _clipPlanePerimeter)
  {
    drawPlanePerimeter(size, vd->pos, clip_plane_point_, clip_plane_normal_, _clipPlaneColor);
  }

  setGLenvironment();

  // Determine texture object extensions:
  for (size_t i = 0; i < 3; ++i)
  {
    // padded borders for (trilinear) interpolation
    size_t paddingLeft = size_t(abs(ptrdiff_t(_visibleRegion.getMin()[i] - _paddingRegion.getMin()[i])));
    size_t paddingRight = size_t(abs(ptrdiff_t(_visibleRegion.getMax()[i] - _paddingRegion.getMax()[i])));
    // a voxels size
    const float vsize = 1.0f / (float)texels[i];
    // half a voxels size
    const float vsize2 = 0.5f / (float)texels[i];
    if (paddingLeft == 0)
    {
      texMin[i] = vsize2;
    }
    else
    {
      texMin[i] = vsize * (float)paddingLeft;
    }

    texMax[i] = (float)vox[i] / (float)texels[i];
    if (paddingRight == 0)
    {
      texMax[i] -= vsize2;
    }
    else
    {
      texMax[i] -= vsize * (float)paddingRight;
    }
  }

  // allow for using raw volume data from vvVolDesc for textures without re-shuffeling
  std::swap(texMin[2], texMax[2]);
  std::swap(texMin[1], texMax[1]);

  // Get OpenGL modelview matrix:
  mat4 mv = gl::getModelviewMatrix();

  renderTex3DPlanar(mv);

  unsetGLenvironment();

  if (_fpsDisplay)
  {
    // Make sure rendering is done to measure correct time.
    // Since this operation is costly, only do it if necessary.
    glFinish();
  }

  vvDebugMsg::msg(3, "vvTexRend::renderVolumeGL() done");
}

//----------------------------------------------------------------------------
/** Activate the previously set clipping plane.
    Clipping plane parameters have to be set with setClippingPlane().
*/
void vvTexRend::activateClippingPlane()
{
  GLdouble planeEq[4];                            // plane equation
  vvVector3 clipNormal2;                          // clipping normal pointing to opposite direction
  float thickness;                                // thickness of single slice clipping plane

  vvDebugMsg::msg(3, "vvTexRend::activateClippingPlane()");

  // Generate OpenGL compatible clipping plane parameters:
  // normal points into oppisite direction
  planeEq[0] = -clip_plane_normal_[0];
  planeEq[1] = -clip_plane_normal_[1];
  planeEq[2] = -clip_plane_normal_[2];
  planeEq[3] = dot( clip_plane_normal_, clip_plane_point_ );
  glClipPlane(GL_CLIP_PLANE0, planeEq);
  glEnable(GL_CLIP_PLANE0);

  // Generate second clipping plane in single slice mode:
  if (_clipSingleSlice)
  {
    thickness = vd->_scale * vd->dist[0] * (vd->vox[0] * 0.01f);
    clipNormal2 = -clip_plane_normal_;
    planeEq[0] = -clipNormal2[0];
    planeEq[1] = -clipNormal2[1];
    planeEq[2] = -clipNormal2[2];
    planeEq[3] = clipNormal2.dot(clip_plane_point_) + thickness;
    glClipPlane(GL_CLIP_PLANE1, planeEq);
    glEnable(GL_CLIP_PLANE1);
  }
}

//----------------------------------------------------------------------------
/** Deactivate the clipping plane.
 */
void vvTexRend::deactivateClippingPlane()
{
  vvDebugMsg::msg(3, "vvTexRend::deactivateClippingPlane()");
  glDisable(GL_CLIP_PLANE0);
  if (_clipSingleSlice) glDisable(GL_CLIP_PLANE1);
}

//----------------------------------------------------------------------------
/** Set number of lights in the scene.
  Fixed material characteristics are used with each setting.
  @param numLights  number of lights in scene (0=ambient light only)
*/
void vvTexRend::setNumLights(const int numLights)
{
  const float ambient[]  = {0.5f, 0.5f, 0.5f, 1.0f};
  const float pos0[] = {0.0f, 10.0f, 10.0f, 0.0f};
  const float pos1[] = {0.0f, -10.0f, -10.0f, 0.0f};

  vvDebugMsg::msg(1, "vvTexRend::setNumLights()");

  // Generate light source 1:
  glEnable(GL_LIGHT0);
  glLightfv(GL_LIGHT0, GL_POSITION, pos0);
  glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);

  // Generate light source 2:
  glEnable(GL_LIGHT1);
  glLightfv(GL_LIGHT1, GL_POSITION, pos1);
  glLightfv(GL_LIGHT1, GL_AMBIENT, ambient);

  // At least 2 lights:
  if (numLights >= 2)
    glEnable(GL_LIGHT1);
  else
    glDisable(GL_LIGHT1);

  // At least one light:
  if (numLights >= 1)
    glEnable(GL_LIGHT0);
  else                                            // no lights selected
    glDisable(GL_LIGHT0);
}

//----------------------------------------------------------------------------
/// @return true if classification is done in no time
bool vvTexRend::instantClassification() const
{
  vvDebugMsg::msg(3, "vvTexRend::instantClassification()");
  return (voxelType != VV_RGBA);
}

//----------------------------------------------------------------------------
/// Returns the number of entries in the RGBA lookup table.
size_t vvTexRend::getLUTSize(vvsize3& size) const
{
  size_t x, y, z;

  vvDebugMsg::msg(3, "vvTexRend::getLUTSize()");
  if (_currentShader==ShaderPreInt && voxelType==VV_PIX_SHD)
  {
    x = y = getPreintTableSize();
    z = 1;
  }
  else
  {
    x = 256;
    if (vd->chan == 2 && vd->tf.size() == 1)
    {
       y = x;
       z = 1;
    }
    else
       y = z = 1;
  }

  size[0] = x;
  size[1] = y;
  size[2] = z;

  return x * y * z;
}

//----------------------------------------------------------------------------
/// Returns the size (width and height) of the pre-integration lookup table.
size_t vvTexRend::getPreintTableSize() const
{
  vvDebugMsg::msg(1, "vvTexRend::getPreintTableSize()");
  return 256;
}

//----------------------------------------------------------------------------
/** Update the color/alpha look-up table.
 Note: glColorTableSGI can have a maximum width of 1024 RGBA entries on IR2 graphics!
 @param dist  slice distance relative to 3D texture sample point distance
              (1.0 for original distance, 0.0 for all opaque).
*/
void vvTexRend::updateLUT(const float dist)
{
  vvDebugMsg::msg(3, "Generating texture LUT. Slice distance = ", dist);

  vvVector4f corr;                                // gamma/alpha corrected RGBA values [0..1]
  vvsize3 lutSize;                                // number of entries in the RGBA lookup table
  lutDistance = dist;
  size_t total = 0;

  if(usePreIntegration)
  {
    glBindTexture(GL_TEXTURE_2D, pixLUTName[0]);
    vd->tf[0].makePreintLUTCorrect(getPreintTableSize(), preintTable, dist);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, getPreintTableSize(), getPreintTableSize(), 0,
        GL_RGBA, GL_UNSIGNED_BYTE, preintTable);
  }
  else
  {
    total = getLUTSize(lutSize);
    if (rgbaTF.size() != rgbaLUT.size())
    {
      rgbaLUT.resize(rgbaTF.size());
    }
    if (pixLUTName.size() > rgbaLUT.size())
    {
       glDeleteTextures(pixLUTName.size()-rgbaLUT.size(), &pixLUTName[rgbaLUT.size()]);
       pixLUTName.resize(rgbaLUT.size());
    }
    else if (pixLUTName.size() < rgbaLUT.size())
    {
       pixLUTName.resize(rgbaLUT.size());
       glGenTextures(pixLUTName.size()-rgbaLUT.size(), &pixLUTName[rgbaLUT.size()]);
    }
    for (size_t chan=0; chan<rgbaTF.size(); ++chan)
    {
      assert(total*4 == rgbaTF[chan].size());
      if (rgbaLUT[chan].size() != rgbaTF[chan].size())
        rgbaLUT[chan].resize(rgbaTF[chan].size());
      for (size_t i=0; i<total; ++i)
      {
        // Gamma correction:
        if (_gammaCorrection)
        {
          corr[0] = gammaCorrect(rgbaTF[chan][i * 4],     VV_RED);
          corr[1] = gammaCorrect(rgbaTF[chan][i * 4 + 1], VV_GREEN);
          corr[2] = gammaCorrect(rgbaTF[chan][i * 4 + 2], VV_BLUE);
          corr[3] = gammaCorrect(rgbaTF[chan][i * 4 + 3], VV_ALPHA);
        }
        else
        {
          corr[0] = rgbaTF[chan][i * 4];
          corr[1] = rgbaTF[chan][i * 4 + 1];
          corr[2] = rgbaTF[chan][i * 4 + 2];
          corr[3] = rgbaTF[chan][i * 4 + 3];
        }

        // Opacity correction:
        // for 0 distance draw opaque slices
        if (dist<=0.0 || (_clipMode == 1 && _clipOpaque)) corr[3] = 1.0f;
        else if (_opacityCorrection) corr[3] = 1.0f - powf(1.0f - corr[3], dist);

        // Convert float to uint8_t and copy to rgbaLUT array:
        for (size_t c=0; c<4; ++c)
        {
          rgbaLUT[chan][i * 4 + c] = uint8_t(corr[c] * 255.0f);
        }
      }

      // Copy LUT to graphics card:
      vvGLTools::printGLError("enter updateLUT()");
      switch (voxelType)
      {
      case VV_RGBA:
        break;
      case VV_PIX_SHD:
        glBindTexture(GL_TEXTURE_2D, pixLUTName[chan]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, lutSize[0], lutSize[1], 0,
            GL_RGBA, GL_UNSIGNED_BYTE, &rgbaLUT[chan][0]);
        break;
      default: assert(0); break;
      }
    }

    if (voxelType == VV_RGBA)
      makeTextures(false);// this mode doesn't use a hardware LUT, so every voxel has to be updated
  }
  vvGLTools::printGLError("leave updateLUT()");
}

//----------------------------------------------------------------------------
/** Set user's viewing direction.
  This information is needed to correctly orientate the texture slices
  in 3D texturing mode if the user is inside the volume.
  @param vd  viewing direction in object coordinates
*/
void vvTexRend::setViewingDirection(vec3f const& vd)
{
  vvDebugMsg::msg(3, "vvTexRend::setViewingDirection()");
  viewDir = vd;
}

//----------------------------------------------------------------------------
/** Set the direction from the viewer to the object.
  This information is needed to correctly orientate the texture slices
  in 3D texturing mode if the viewer is outside of the volume.
  @param vd  object direction in object coordinates
*/
void vvTexRend::setObjectDirection(vec3f const& od)
{
  vvDebugMsg::msg(3, "vvTexRend::setObjectDirection()");
  objDir = od;
}


bool vvTexRend::checkParameter(ParameterType param, vvParam const& value) const
{
  switch (param)
  {
  case VV_SLICEINT:

    {
      virvo::tex_filter_mode mode = static_cast< virvo::tex_filter_mode >(value.asInt());

      if (mode == virvo::Nearest || mode == virvo::Linear)
      {
        return true;
      }
    }

    return false;;

  default:

    return vvRenderer::checkParameter(param, value);

  }
}


//----------------------------------------------------------------------------
// see parent
void vvTexRend::setParameter(ParameterType param, const vvParam& newValue)
{
  vvDebugMsg::msg(3, "vvTexRend::setParameter()");
  switch (param)
  {
    case vvRenderer::VV_GAMMA:
      // fall-through
    case vvRenderer::VV_GAMMA_CORRECTION:
      vvRenderer::setParameter(param, newValue);
      updateTransferFunction();
      break;
    case vvRenderer::VV_SLICEINT:
      if (_interpolation != static_cast< virvo::tex_filter_mode >(newValue.asInt()))
      {
        _interpolation = static_cast< virvo::tex_filter_mode >(newValue.asInt());
        for (size_t f = 0; f < vd->frames; ++f)
        {
          glBindTexture(GL_TEXTURE_3D_EXT, texNames[f]);
          glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
          glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
        }
        updateTransferFunction();
      }
      break;
    case vvRenderer::VV_MIN_SLICE:
      minSlice = newValue;
      break;
    case vvRenderer::VV_MAX_SLICE:
      maxSlice = newValue;
      break;
    case vvRenderer::VV_SLICEORIENT:
      _sliceOrientation = (SliceOrientation)newValue.asInt();
      break;
    case vvRenderer::VV_PREINT:
      _preIntegration = newValue;
      updateTransferFunction();
      disableShader(_shader);
      delete _shader;
      _shader = initShader();
      break;
    case vvRenderer::VV_BINNING:
      vd->_binning = (vvVolDesc::BinningType)newValue.asInt();
      break;
    case vvRenderer::VV_OFFSCREENBUFFER:
    case vvRenderer::VV_USE_OFFSCREEN_BUFFER:
      {
        bool fbo = static_cast<bool>(newValue);

        this->_useOffscreenBuffer = fbo;

        if (fbo)
          setRenderTarget( virvo::FramebufferObjectRT::create() );
        else
          setRenderTarget( virvo::NullRT::create() );
      }
      break;
    case vvRenderer::VV_IMG_SCALE:
      //_imageScale = newValue;
      break;
    case vvRenderer::VV_IMG_PRECISION:
    case vvRenderer::VV_IMAGE_PRECISION:
      {
        virvo::BufferPrecision bp = mapBitsToBufferPrecision(static_cast<int>(newValue));

        this->_imagePrecision = bp;

//      setRenderTarget( virvo::FramebufferObjectRT::create(mapBufferPrecisionToFormat(bp), virvo::PF_DEPTH32F_STENCIL8) );
        setRenderTarget( virvo::FramebufferObjectRT::create(mapBufferPrecisionToFormat(bp), virvo::PF_DEPTH24_STENCIL8) );
      }
      break;
    case vvRenderer::VV_LIGHTING:
      if (newValue.asBool())
      {
        _previousShader = _currentShader;
          _currentShader = ShaderLighting;
      }
      else
      {
        _currentShader = _previousShader;
      }
      disableShader(_shader);
      delete _shader;
      _shader = initShader();
      break;
    case vvRenderer::VV_PIX_SHADER:
      setCurrentShader(newValue);
      break;
    case vvRenderer::VV_PADDING_REGION:
      vvRenderer::setParameter(param, newValue);
      makeTextures(true);
      break;
    default:
      vvRenderer::setParameter(param, newValue);
      break;
  }
}

//----------------------------------------------------------------------------
// see parent for comments
vvParam vvTexRend::getParameter(ParameterType param) const
{
  vvDebugMsg::msg(3, "vvTexRend::getParameter()");

  switch (param)
  {
    case vvRenderer::VV_MIN_SLICE:
      return minSlice;
    case vvRenderer::VV_MAX_SLICE:
      return maxSlice;
    case vvRenderer::VV_SLICEORIENT:
      return (int)_sliceOrientation;
    case vvRenderer::VV_BINNING:
      return (int)vd->_binning;
    case vvRenderer::VV_PIX_SHADER:
      return getCurrentShader();
    default:
      return vvRenderer::getParameter(param);
  }
}

//----------------------------------------------------------------------------
/** Get information on hardware support for rendering modes.
  @param geom voxel type to get information about
  @return true if the requested voxel type is supported by
    the system's graphics hardware.
*/
bool vvTexRend::isSupported(const VoxelType voxel)
{
  vvDebugMsg::msg(3, "vvTexRend::isSupported(1)");

  switch(voxel)
  {
    case VV_BEST:
    case VV_RGBA:
      return true;
    case VV_PIX_SHD:
      {
        return (vvShaderFactory::isSupported("cg")
          || vvShaderFactory::isSupported("glsl"));
      }
    default: return false;
  }
}

//----------------------------------------------------------------------------
/** Return true if a feature is supported.
 */
bool vvTexRend::isSupported(const FeatureType feature) const
{
  vvDebugMsg::msg(3, "vvTexRend::isSupported()");
  switch(feature)
  {
    case VV_MIP: return true;
    default: assert(0); break;
  }
  return false;
}

//----------------------------------------------------------------------------
/** Return the currently used voxel type.
  This is expecially useful if VV_AUTO was passed in the constructor.
*/
vvTexRend::VoxelType vvTexRend::getVoxelType() const
{
  vvDebugMsg::msg(3, "vvTexRend::getVoxelType()");
  return voxelType;
}

//----------------------------------------------------------------------------
/** Return the currently used pixel shader [0..numShaders-1].
 */
int vvTexRend::getCurrentShader() const
{
  vvDebugMsg::msg(3, "vvTexRend::getCurrentShader()");
  return _currentShader;
}

//----------------------------------------------------------------------------
/** Set the currently used pixel shader [0..numShaders-1].
 */
void vvTexRend::setCurrentShader(const int shader)
{
  vvDebugMsg::msg(3, "vvTexRend::setCurrentShader()");
  if(shader >= NUM_PIXEL_SHADERS || shader < 0)
    _currentShader = Shader1Chan;
  else
    _currentShader = shader;

  disableShader(_shader);
  delete _shader;
  _shader = initShader();
}

//----------------------------------------------------------------------------
/// inherited from vvRenderer, only valid for planar textures
void vvTexRend::renderQualityDisplay() const
{
  const int numSlices = int(_quality * 100.0f);
  vvPrintGL printGL;
  vec4f clearColor = vvGLTools::queryClearColor();
  vec4f fontColor( 1.0f - clearColor[0], 1.0f - clearColor[1], 1.0f - clearColor[2], 1.0f );
  printGL.setFontColor(fontColor);
  printGL.print(-0.9f, 0.9f, "Textures: %d", numSlices);
}

//----------------------------------------------------------------------------
void vvTexRend::enableTexture(const GLenum target) const
{
  if (voxelType != VV_PIX_SHD)
    glEnable(target);
}

//----------------------------------------------------------------------------
void vvTexRend::disableTexture(const GLenum target) const
{
  if (voxelType != VV_PIX_SHD)
    glDisable(target);
}

//----------------------------------------------------------------------------
void vvTexRend::enableShader(vvShaderProgram* shader) const
{
  vvGLTools::printGLError("Enter vvTexRend::enableShader()");

  if(!shader)
    return;

  shader->enable();

  if(VV_PIX_SHD == voxelType)
  {
    if (_currentShader == ShaderMultiTF)
    {
      shader->setParameter1i("channels", vd->chan);
      for (size_t chan=0; chan < pixLUTName.size(); ++chan)
      {
        std::stringstream str;
        str << "pixLUT" << chan;
        shader->setParameterTex2D(str.str().c_str(), pixLUTName[chan]);
      }
    }
    else
    {
      shader->setParameterTex2D("pixLUT", pixLUTName[0]);
    }

    if (_channel4Color != NULL)
    {
      shader->setParameter3f("chan4color", _channel4Color[0], _channel4Color[1], _channel4Color[2]);
    }
    if (_opacityWeights != NULL)
    {
      shader->setParameter4f("opWeights", _opacityWeights[0], _opacityWeights[1], _opacityWeights[2], _opacityWeights[3]);
    }
  }

  vvGLTools::printGLError("Leaving vvTexRend::enablePixelShaders()");
}

//----------------------------------------------------------------------------
void vvTexRend::disableShader(vvShaderProgram* shader) const
{
  vvGLTools::printGLError("Enter vvTexRend::disableShader()");

  if (shader)
  {
    shader->disable();
  }

  vvGLTools::printGLError("Leaving vvTexRend::disableShader()");
}

void vvTexRend::initClassificationStage()
{
  if(voxelType==VV_PIX_SHD)
  {
    pixLUTName.resize(vd->tf.size());
    glGenTextures(pixLUTName.size(), &pixLUTName[0]);
  }
}

void vvTexRend::freeClassificationStage()
{
  if (voxelType==VV_PIX_SHD)
  {
    glDeleteTextures(pixLUTName.size(), &pixLUTName[0]);
  }
}


//----------------------------------------------------------------------------
/** @return Pointer of initialized ShaderProgram or NULL
 */
vvShaderProgram* vvTexRend::initShader()
{
  vvGLTools::printGLError("Enter vvTexRend::initShader()");

  std::ostringstream fragName;
  if(voxelType == VV_PIX_SHD)
  {
    fragName << "shader" << std::setw(2) << std::setfill('0') << (_currentShader+1);
  }

  // intersection on CPU, try to create fragment program
  vvShaderProgram* shader = _shaderFactory->createProgram("", "", fragName.str());

  vvGLTools::printGLError("Leave vvTexRend::initShader()");

  return shader;
}

//----------------------------------------------------------------------------
void vvTexRend::printLUT(size_t chan) const
{
  vvsize3 lutEntries;

  size_t total = getLUTSize(lutEntries);
  for (size_t i=0; i<total; ++i)
  {
    cerr << "#" << i << ": ";
    for (size_t c=0; c<4; ++c)
    {
      cerr << int(rgbaLUT[chan][i * 4 + c]);
      if (c<3) cerr << ", ";
    }
    cerr << endl;
  }
}

uint8_t* vvTexRend::getHeightFieldData(float points[4][3], size_t& width, size_t& height)
{
  GLint viewport[4];
  uint8_t *pixels, *data, *result=NULL;
  size_t numPixels;
  size_t index;
  float sizeX, sizeY;
  vvVector3 size, size2;
  vvVector3 texcoord[4];

  std::cerr << "getHeightFieldData" << endl;

  glGetIntegerv(GL_VIEWPORT, viewport);

  width = size_t(ceil(getManhattenDist(points[0], points[1])));
  height = size_t(ceil(getManhattenDist(points[0], points[3])));

  numPixels = width * height;
  pixels = new uint8_t[4*numPixels];

  glReadPixels(viewport[0], viewport[1], width, height,
    GL_RGBA, GL_UNSIGNED_BYTE, pixels);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();

  size = vd->getSize();
  for (size_t i = 0; i < 3; ++i)
    size2[i]   = 0.5f * size[i];

  for (size_t j = 0; j < 4; j++)
    for (size_t k = 0; k < 3; k++)
  {
    texcoord[j][k] = (points[j][k] + size2[k]) / size[k];
    texcoord[j][k] = texcoord[j][k] * (texMax[k] - texMin[k]) + texMin[k];
  }

  enableTexture(GL_TEXTURE_3D_EXT);
  glBindTexture(GL_TEXTURE_3D_EXT, texNames[vd->getCurrentFrame()]);

  if (glIsTexture(texNames[vd->getCurrentFrame()]))
    std::cerr << "true" << endl;
  else
    std::cerr << "false" << endl;

  sizeX = 2.0f * float(width)  / float(viewport[2] - 1);
  sizeY = 2.0f * float(height) / float(viewport[3] - 1);

  std::cerr << "SizeX: " << sizeX << endl;
  std::cerr << "SizeY: " << sizeY << endl;
  std::cerr << "Viewport[2]: " << viewport[2] << endl;
  std::cerr << "Viewport[3]: " << viewport[3] << endl;

  std::cerr << "TexCoord1: " << texcoord[0][0] << " " << texcoord[0][1] << " " << texcoord[0][2] << endl;
  std::cerr << "TexCoord2: " << texcoord[1][0] << " " << texcoord[1][1] << " " << texcoord[1][2] << endl;
  std::cerr << "TexCoord3: " << texcoord[2][0] << " " << texcoord[2][1] << " " << texcoord[2][2] << endl;
  std::cerr << "TexCoord4: " << texcoord[3][0] << " " << texcoord[3][1] << " " << texcoord[3][2] << endl;

  glBegin(GL_QUADS);
  glTexCoord3f(texcoord[0][0], texcoord[0][1], texcoord[0][2]);
  glVertex3f(-1.0, -1.0, -1.0);
  glTexCoord3f(texcoord[1][0], texcoord[1][1], texcoord[1][2]);
  glVertex3f(sizeX, -1.0, -1.0);
  glTexCoord3f(texcoord[2][0], texcoord[2][1], texcoord[2][2]);
  glVertex3f(sizeX, sizeY, -1.0);
  glTexCoord3f(texcoord[3][0], texcoord[3][1], texcoord[3][2]);
  glVertex3f(-1.0, sizeY, -1.0);
  glEnd();

  glFinish();
  glReadBuffer(GL_BACK);

  data = new uint8_t[texelsize * numPixels];
  memset(data, 0, texelsize * numPixels);
  glReadPixels(viewport[0], viewport[1], width, height,
    GL_RGB, GL_UNSIGNED_BYTE, data);

  std::cerr << "data read" << endl;

  if (vd->chan == 1 && (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4))
  {
    result = new uint8_t[numPixels];
    for (size_t y = 0; y < height; y++)
      for (size_t x = 0; x < width; x++)
    {
      index = y * width + x;
      switch (voxelType)
      {
        case VV_PIX_SHD:
          result[index] = data[texelsize*index];
          break;
        case VV_RGBA:
          assert(0);
          break;
        default:
          assert(0);
          break;
      }
      std::cerr << "Result: " << index << " " << (int) (result[index]) << endl;
    }
  }
  else if (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4)
  {
    if ((voxelType == VV_RGBA) || (voxelType == VV_PIX_SHD))
    {
      result = new uint8_t[vd->chan * numPixels];

      for (size_t y = 0; y < height; y++)
        for (size_t x = 0; x < width; x++)
      {
        index = (y * width + x) * vd->chan;
        for (size_t c = 0; c < vd->chan; c++)
        {
          result[index + c] = data[index + c];
          std::cerr << "Result: " << index+c << " " << (int) (result[index+c]) << endl;
        }
      }
    }
    else
      assert(0);
  }

  std::cerr << "result read" << endl;

  disableTexture(GL_TEXTURE_3D_EXT);

  glPopMatrix();

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

  return result;
}

float vvTexRend::getManhattenDist(float p1[3], float p2[3]) const
{
  float dist = 0;

  for (size_t i=0; i<3; ++i)
  {
    dist += float(fabs(p1[i] - p2[i])) / float(vd->getSize()[i] * vd->vox[i]);
  }

  std::cerr << "Manhattan Distance: " << dist << endl;

  return dist;
}

void vvTexRend::evaluateLocalIllumination(vvShaderProgram* shader, const vvVector3& normal)
{
  // Local illumination based on blinn-phong shading.
  if (voxelType == VV_PIX_SHD && _currentShader == ShaderLighting)
  {
    // Light direction.
    const vvVector3 L(-normal);

    // Viewing direction.
    const vvVector3 V(-normal);

    // Half way vector.
    vvVector3 H(L + V);
    H.normalize();
    shader->setParameter3f("L", L[0], L[1], L[2]);
    shader->setParameter3f("H", H[0], H[1], H[2]);
  }
}

size_t vvTexRend::getTextureSize(size_t sz) const
{
  if (extNonPower2)
    return sz;

  return vvToolshed::getTextureSize(sz);
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
