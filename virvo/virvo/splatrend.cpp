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

#include <GL/glew.h>

#include "vvclock.h"
#include "vvshaderfactory.h"
#include "splatrend.h"
#include "vvtoolshed.h"
#include "vvvoldesc.h"

#include "math/math.h"

#include "private/vvgltools.h"
#include "private/vvlog.h"

#include "cl/utils.h"
#include "gl/util.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef __APPLE__
#include <OpenCL/OpenCL.h>
#include <OpenGL/OpenGL.h>
#else
#include <CL/cl.h>
#include <CL/cl_gl.h>
#endif

#if VV_HAVE_X11
#include <GL/glx.h>
#include <X11/Xlib.h>
#endif

#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>

// TODO: deduce this type from CL_DEVICE_ADDRESS_BITS
typedef uint32_t index_t;

namespace virvo
{

typedef recti Viewport;

}

namespace cl = virvo::cl;
namespace gl = virvo::gl;

namespace virvo
{
namespace cl
{

class buffer
{
public:

    // CL-spec says that cl_mem is a pointer type
    buffer() : data_(nullptr) {}
    /* implicit */ buffer(cl_mem data) : data_(data) {}

    ~buffer()
    {
        cl_int err = clReleaseMemObject(data_);

        if (err != CL_SUCCESS)
        {
            // stderr - don't throw from dtor..
            VV_LOG(0) << "Error releasing memory object: " << virvo::cl::errorString(err);
        }
    }

    void operator=(cl_mem const& data) { data_ = data; }
    operator cl_mem() const { return data_; }

    cl_mem* ptr() { return &data_; }
    cl_mem const* ptr() const { return &data_; }

private:

    cl_mem data_;

};

} // cl
} // virvo

struct CLProgram
{
  CLProgram()
    : clpbo(nullptr)
    , glpbo(0)
    , gltex(0)
    , spheres(nullptr)
    , bvhnodes(nullptr)
    , scalars(nullptr)
    , tf(nullptr)
    , vertexbuffer(nullptr)
    , indexbuffer(nullptr)
    , kdleafindices(nullptr)
  {
  }

  cl_device_id deviceid;
  std::vector<cl_context_properties> ctxproperties;
  cl_context context;
  cl_command_queue commands;
  cl_program program;
  cl_kernel reorderkernel;
  cl_kernel bitonicsortkernel;
  cl_kernel spherekernel;

  cl::buffer clpbo;
  GLuint glpbo;
  GLuint gltex;

  cl::buffer spheres;
  cl::buffer bvhnodes;
  cl::buffer scalars;
  cl::buffer tf;

  cl::buffer vertexbuffer;
  cl::buffer indexbuffer;
  cl::buffer kdleafindices;
};

typedef virvo::basic_aabb< ssize_t > box;

struct KdTreeNode : public box
{
  KdTreeNode(virvo::vector< 3, size_t> const& min, virvo::vector< 3, size_t > const& max)
    : box(min, max)
    , leftChild(nullptr)
    , rightChild(nullptr)
    , parent(nullptr)
  {
  }

    typedef std::shared_ptr< KdTreeNode > node_ptr;
    node_ptr leftChild;
    node_ptr rightChild;
    node_ptr parent;

  /*! \brief  leafs are numbered consecutively for reordering
   */
  index_t firstChildOffset;
  index_t firstSphereOffset;

  bool isLeaf() const
  {
    return leftChild == nullptr && rightChild == nullptr;
  }
};

struct Brick : public box
{
  Brick(virvo::vector< 3, size_t > const& min, virvo::vector< 3, size_t > const& max) : box(min, max), minvox(0), maxvox(0), avg(0) {}
  uint8_t minvox;
  uint8_t maxvox;
  uint8_t avg;
  virvo::vec4 avg_rgba;
};

namespace
{
void createImage(size_t w, size_t h, const std::unique_ptr<CLProgram>& program,
    cl_mem* buffer, const void* data, cl_channel_order co = CL_RGBA)
{
    clReleaseMemObject(*buffer);
    int err;
    cl_image_format format = { co, CL_FLOAT };
#if CL_VERSION_1_2
    cl_image_desc desc;
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = w;
    desc.image_height = h;
    desc.image_depth = 1;
    desc.image_array_size = 1;
    desc.image_row_pitch = 0;
    desc.image_slice_pitch = 0;
    desc.buffer = NULL;
    desc.num_mip_levels = 0;
    desc.num_samples = 0;
    *buffer = clCreateImage(program->context, CL_MEM_READ_ONLY, &format, &desc, NULL, &err);
#else
    *buffer = clCreateImage2D(program->context, CL_MEM_READ_ONLY, &format, w, h, 0, NULL, &err);
#endif
    if (err != CL_SUCCESS)
    {
        VV_LOG(0) << "Error creating image buffer: " << virvo::cl::errorString(err);
        return;
    }

    size_t origin[] = { 0, 0, 0 };
    size_t region[] = { w, h, 1 };

    size_t mul = format.image_channel_order == CL_RGBA ? 4 : 1;
    err = clEnqueueWriteImage(program->commands, *buffer, CL_TRUE, origin, region, w * mul * sizeof(float), 0, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        VV_LOG(0) << "Error copying image to device: " << virvo::cl::errorString(err);
        return;
    }
}

CLProgram* initCLProgram()
{
  std::string clIncludeDir;
#ifdef VV_CL_KERNEL_PATH
  clIncludeDir = VV_CL_KERNEL_PATH;
#endif

  if (const char* incdir = getenv("VV_CL_KERNEL_PATH"))
  {
    clIncludeDir = incdir;
  }

  CLProgram* program = new CLProgram;

  cl_uint pcount;
  clGetPlatformIDs(0, NULL, &pcount);
  std::vector<cl_platform_id> platforms(pcount);
  clGetPlatformIDs(pcount, &platforms[0], NULL);
  if (pcount == 0)
  {
    VV_LOG(0) << "No OpenCL platforms found";
    delete program;
    return nullptr;
  }

  cl_uint dcount;
  cl_int err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &dcount);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "Error creating device group: " << virvo::cl::errorString(err);
    delete program;
    return nullptr;
  }
  if (dcount == 0)
  {
    VV_LOG(0) << "No OpenCL devices found: " << virvo::cl::errorString(err);
    delete program;
    return nullptr;
  }

  std::vector<cl_device_id> deviceids(dcount);
  err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, dcount, &deviceids[0], NULL);

  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "Error enumerating device IDs (dcount=" << dcount << "): " << virvo::cl::errorString(err);
    delete program;
    return nullptr;
  }

  VV_LOG(0) << "Have " << dcount << " OpenCL devices";
  for (cl_uint i=0; i<dcount; ++i)
  {
     std::vector<char> vendor;
     size_t sz = 0;
     err = clGetDeviceInfo(deviceids[i], CL_DEVICE_VENDOR, 0, NULL, &sz);
     if (err == CL_SUCCESS)
     {
        vendor.resize(sz);
        err = clGetDeviceInfo(deviceids[i], CL_DEVICE_VENDOR, vendor.size(), &vendor[0], NULL);
        VV_LOG(0) << "Device " << i << " vendor: " << std::string(&vendor[0], vendor.size());
     }
  }


  // TODO: choose a device
  program->deviceid = deviceids[dcount-1];

#ifdef __APPLE__
  CGLContextObj glContext = CGLGetCurrentContext();
  CGLShareGroupObj shareGroup = CGLGetShareGroup(glContext);

  program->ctxproperties.push_back(CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE);
  program->ctxproperties.push_back((cl_context_properties)shareGroup);
  program->ctxproperties.push_back(0);
#endif

#ifdef __linux__
  program->ctxproperties.push_back(CL_GL_CONTEXT_KHR);
  program->ctxproperties.push_back((cl_context_properties)glXGetCurrentContext());
  program->ctxproperties.push_back(CL_GLX_DISPLAY_KHR);
  program->ctxproperties.push_back((cl_context_properties)glXGetCurrentDisplay());
  program->ctxproperties.push_back(CL_CONTEXT_PLATFORM);
  program->ctxproperties.push_back((cl_context_properties)platforms[0]);
  program->ctxproperties.push_back(0);
#endif

#ifdef _WIN32
  program->ctxproperties.push_back(CL_GL_CONTEXT_KHR);
  program->ctxproperties.push_back((cl_context_properties)wglGetCurrentContext());
  program->ctxproperties.push_back(CL_WGL_HDC_KHR);
  program->ctxproperties.push_back((cl_context_properties)wglGetCurrentDC());
  program->ctxproperties.push_back(CL_CONTEXT_PLATFORM);
  program->ctxproperties.push_back((cl_context_properties)platforms[0]);
  program->ctxproperties.push_back(0);
#endif

  program->context = clCreateContext(&program->ctxproperties[0], 1, &program->deviceid, NULL, NULL, &err);
  if (!program->context)
  {
    VV_LOG(0) << "Error creating compute context: " << virvo::cl::errorString(err);
    delete program;
    return nullptr;
  }

  program->commands = clCreateCommandQueue(program->context, program->deviceid, CL_QUEUE_PROFILING_ENABLE, &err);
  if (!program->commands)
  {
    VV_LOG(0) << "Error creating command queue: " << virvo::cl::errorString(err);
    delete program;
    return nullptr;
  }

  const static std::string filename = "spheres.cl";
//  const static std::string filename = "kdsort.cl";
  std::stringstream kernelpathstr;
  kernelpathstr << clIncludeDir << filename;
  std::string kernelstr = vvToolshed::file2string(kernelpathstr.str());

  const char* source = kernelstr.c_str();
  program->program = clCreateProgramWithSource(program->context, 1, (const char**)&source, NULL, &err);
  if (!program->program)
  {
    VV_LOG(0) << "Error creating program: " << virvo::cl::errorString(err);
    delete program;
    return nullptr;
  }

  std::stringstream optstr;
  optstr << "-I";
  optstr << clIncludeDir;

  optstr << " ";
  optstr << "-DMAXTEXWIDTH=";
  size_t maximgw;
  clGetDeviceInfo(program->deviceid, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(maximgw), &maximgw, NULL);
  optstr << maximgw;

  optstr << " ";
  optstr << "-DMAXTEXWIDTHLOG2=";
  optstr << log2(maximgw);

  std::string options = optstr.str();
  err = clBuildProgram(program->program, 0, NULL, options.c_str(), NULL, NULL);
  if (err != CL_SUCCESS)
  {
    size_t len;
    char buf[2048];
    VV_LOG(0) << "Error building program. Log:";
    clGetProgramBuildInfo(program->program, program->deviceid, CL_PROGRAM_BUILD_LOG, sizeof(buf), buf, &len);
    VV_LOG(0) << buf;
    delete program;
    return nullptr;
  }

  std::string kernelname = "trace";
  program->spherekernel = clCreateKernel(program->program, kernelname.c_str(), &err);
  if (!program->spherekernel || err != CL_SUCCESS)
  {
    VV_LOG(0) << "Error creating sphere kernel: " << virvo::cl::errorString(err);
    delete program;
    return nullptr;
  }

  // init gl pbo for cl/gl interop

  virvo::Viewport vp = gl::getViewport();
  unsigned int texels = vvToolshed::getTextureSize(vp[2]) * vvToolshed::getTextureSize(vp[3]);
  unsigned int numvalues = texels * 4;
  unsigned int size = sizeof(GLfloat) * numvalues;

  glGenBuffers(1, &program->glpbo);
  glBindBuffer(GL_ARRAY_BUFFER, program->glpbo);
  glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glGenTextures(1, &program->gltex);
  glBindTexture(GL_TEXTURE_2D, program->gltex);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, vvToolshed::getTextureSize(vp[2]),
    vvToolshed::getTextureSize(vp[3]), 0, GL_RGBA, GL_FLOAT, NULL);

  program->clpbo = clCreateFromGLBuffer(program->context, CL_MEM_READ_WRITE, program->glpbo, &err);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "clCreateFromGLBuffer() failed: " << virvo::cl::errorString(err);
    delete program;
    return nullptr;
  }

  return program;
}

void updateCLPoints(const std::unique_ptr<CLProgram>& program, const std::vector<float>& bvhnodes,
  const std::vector<cl_float4>& spheres, const std::vector<float>& scalars)
{
  size_t maximgw;
  clGetDeviceInfo(program->deviceid, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(maximgw), &maximgw, NULL);

  size_t bnw = virvo::toolshed::iDivUp<size_t>(bvhnodes.size(), 4) < maximgw ? virvo::toolshed::iDivUp<size_t>(bvhnodes.size(), 4) : maximgw;
  size_t bnh = virvo::toolshed::iDivUp<size_t>(virvo::toolshed::iDivUp<size_t>(bvhnodes.size(), 4), maximgw);
  createImage(bnw, bnh, program, program->bvhnodes.ptr(), bvhnodes.data());

  size_t sw = spheres.size() < maximgw ? spheres.size() : maximgw;
  size_t sh = virvo::toolshed::iDivUp(spheres.size(), maximgw);
  createImage(sw, sh, program, program->spheres.ptr(), spheres.data());

  size_t scw = scalars.size() < maximgw ? scalars.size() : maximgw;
  size_t sch = virvo::toolshed::iDivUp(scalars.size(), maximgw);
  createImage(scw, sch, program, program->scalars.ptr(), scalars.data(), CL_R);
}

void updateCLTf(const std::unique_ptr<CLProgram>& program, const std::vector<float>& tf, float scale)
{
  size_t maximgw;
  clGetDeviceInfo(program->deviceid, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(maximgw), &maximgw, NULL);

  std::vector<cl_float4> tf4;
  for (size_t i = 0; i < tf.size(); i += 4)
  {
    cl_float4 clr = { tf[i], tf[i + 1], tf[i + 2], tf[i + 3] * scale };
    clr.s[0] *= 0.7f;
    clr.s[1] *= 0.7f;
    clr.s[2] *= 0.7f;
    clr.s[3] *= 1.3f;
    if (clr.s[3] > 1.0f) clr.s[3] = 1.0f;
    tf4.push_back(clr);
  } 
    
  size_t tfw = tf4.size();
  assert(tfw == 256);
  createImage(tfw, 1, program, program->tf.ptr(), tf4.data());
}

bool sortLeafs(const std::unique_ptr<CLProgram>& program, const std::vector<index_t>& indices,
  size_t pointspernode, GLuint* indexbuffer)
{
  program->kdleafindices = clCreateBuffer(program->context, CL_MEM_READ_ONLY, sizeof(index_t) * indices.size(), NULL, NULL);
  cl_int err = clEnqueueWriteBuffer(program->commands, program->kdleafindices, CL_TRUE, 0, sizeof(index_t) * indices.size(), indices.data(), 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "clEnqueueWriteBuffer(indices) failed";
    return false;
  }

  program->indexbuffer = clCreateFromGLBuffer(program->context, 0, *indexbuffer, &err);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "clCreateFromGLBuffer(indexbuffer) failed: " << virvo::cl::errorString(err);
    return false;
  }

  size_t numleafindices = indices.size();

  size_t localWorkSize[1] = { 256 };
  size_t globalWorkSize[1] = { vvToolshed::getTextureSize(indices.size() * pointspernode) };

  cl_uint arg = 0;
  err = 0;
  err |= clSetKernelArg(program->reorderkernel, arg++, sizeof(cl_mem), &program->indexbuffer);
  err |= clSetKernelArg(program->reorderkernel, arg++, sizeof(cl_mem), &program->kdleafindices);
  err |= clSetKernelArg(program->reorderkernel, arg++, sizeof(numleafindices), &numleafindices);
  err |= clSetKernelArg(program->reorderkernel, arg++, sizeof(pointspernode), &pointspernode);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "clSetKernelArg() failed";
    return false;
  }

  err = clEnqueueAcquireGLObjects(program->commands, 1, program->indexbuffer.ptr(), 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "clEnqueueAcquireGLObjects(indexbuffer) failed";
    return false;
  }

  err = clEnqueueNDRangeKernel(program->commands, program->reorderkernel, 1, NULL, &globalWorkSize[0], &localWorkSize[0], 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "clEnqueueNDRangeKernel() failed. Error code: " << virvo::cl::errorString(err);
    return false;
  }

  clFinish(program->commands);

  err = clEnqueueReleaseGLObjects(program->commands, 1, program->indexbuffer.ptr(), 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "clEnqueueReleaseGLObjects(indexbuffer) failed";
    return false;
  }

  return true;
}

bool sortPointsInLeafs(const std::unique_ptr<CLProgram>& program, const std::vector<index_t>& indices,
  size_t pointspernode, GLuint* vertexbuffer, GLuint* indexbuffer, virvo::vec3 const& eye)
{
  cl_int err = CL_SUCCESS;
  program->vertexbuffer = clCreateFromGLBuffer(program->context, 0, *vertexbuffer, &err);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "clCreateFromGLBuffer(vertexbuffer) failed";
    return false;
  }

  program->indexbuffer = clCreateFromGLBuffer(program->context, 0, *indexbuffer, &err);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "clCreateFromGLBuffer(indexbuffer) failed";
    return false;
  }

  size_t localWorkSize[1] = { 256 };
  size_t globalWorkSize[1] = { vvToolshed::getTextureSize(indices.size() * pointspernode) };

  cl_float4 veye = { eye[0], eye[1], eye[2] };
  index_t iterations = static_cast<index_t>(pointspernode / localWorkSize[0]);

  cl_uint arg = 0;
  err = 0;
  err |= clSetKernelArg(program->bitonicsortkernel, arg++, sizeof(cl_mem), &program->vertexbuffer);
  err |= clSetKernelArg(program->bitonicsortkernel, arg++, sizeof(cl_mem), &program->indexbuffer);
  err |= clSetKernelArg(program->bitonicsortkernel, arg++, sizeof(uint) * 256, NULL);
  err |= clSetKernelArg(program->bitonicsortkernel, arg++, sizeof(veye), &veye);
  err |= clSetKernelArg(program->bitonicsortkernel, arg++, sizeof(iterations), &iterations);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "clSetKernelArg() failed";
    return false;
  }

  err = clEnqueueAcquireGLObjects(program->commands, 1, program->vertexbuffer.ptr(), 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "clEnqueueAcquireGLObjects(vertexbuffer) failed";
    return false;
  }

  err = clEnqueueAcquireGLObjects(program->commands, 1, program->indexbuffer.ptr(), 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "clEnqueueAcquireGLObjects(indexbuffer) failed";
    return false;
  }

  err = clEnqueueNDRangeKernel(program->commands, program->bitonicsortkernel, 1, NULL, &globalWorkSize[0], &localWorkSize[0], 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "clEnqueueNDRangeKernel() failed. Error code: " << virvo::cl::errorString(err);
    return false;
  }

  clFinish(program->commands);

  err = clEnqueueReleaseGLObjects(program->commands, 1, program->vertexbuffer.ptr(), 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "clEnqueueReleaseGLObjects(vertexbuffer) failed";
    return false;
  }

  err = clEnqueueReleaseGLObjects(program->commands, 1, program->indexbuffer.ptr(), 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "clEnqueueReleaseGLObjects(indexbuffer) failed";
    return false;
  }

  return true;
}


void genPoints(std::vector< virvo::vec4 >* points, std::vector< virvo::vec4 >* texcoords,
  std::vector<float>* scalars,
  std::vector<GLuint>* indices, const std::vector< std::shared_ptr< KdTreeNode > >& kdleafs,
  size_t numpernode, const vvVolDesc* vd)
{
  typedef std::minstd_rand0 random_number_generator;
  random_number_generator randgen(0);

  virvo::vec3 size = vd->getSize();

  uint8_t* raw = vd->getRaw(vd->getCurrentFrame());

  GLuint idx = 0;
  for (auto it = kdleafs.begin(); it != kdleafs.end(); ++it)
  {
    assert((*it)->leftChild == nullptr && (*it)->rightChild == nullptr);
    virvo::aabb node(vd->objectCoords((*it)->min), vd->objectCoords((*it)->max));

    (*it)->firstSphereOffset = idx;
    for (size_t pts = 0; pts < numpernode; ++pts)
    {
      virvo::vec4 point;
      virvo::vec4 texcoord;
      virvo::vector< 3, size_t > texcoordi;
      for (size_t d = 0; d < 3; ++d)
      {
        double r = (double)randgen() / (double)random_number_generator::max();
        point[d] = static_cast<float>(r) * (node.max[d] - node.min[d]);
        point[d] += node.min[d];
        texcoord[d] = point[d];
        texcoord[d] /= size[d];
        texcoord[d] += 0.5f;
        texcoord[d] = ts_clamp(texcoord[d], 0.0f, 1.0f);
        texcoordi[d] = static_cast<size_t>(texcoord[d] * ((float)vd->vox[d] - 1));
      }
      texcoord[3] = 1.0f;
      point[3] = 1.0f;
      size_t txidx = texcoordi[2] * vd->vox[0] * vd->vox[1] + texcoordi[1] * vd->vox[0] + texcoordi[0];
      float sample = float(raw[txidx]) / 256.0f;
      scalars->push_back(sample);
      points->push_back(point);
      texcoords->push_back(texcoord);
      indices->push_back(idx);
      ++idx;
    }
  }
}

inline std::ostream& operator<<(std::ostream& out, const Brick& b)
{
  out << "Min: " << (int)b.minvox << ", max: " << (int)b.maxvox << ", avg: " << (int)b.avg;
  return out;
}

void initBrick(const vvVolDesc* vd, Brick* b, const std::vector<float>& rgbatf)
{
  const static float threshold = 0.5f;
  uint8_t maxvox = 0;
  uint8_t minvox = std::numeric_limits<uint8_t>::max();
  uint32_t avg = 0;
  virvo::vec4 avg_rgba(0.0f);
  for (ssize_t z = b->min[2]; z < b->max[2]; ++z)
  {
    for (ssize_t y = b->min[1]; y < b->max[1]; ++y)
    {
      for (ssize_t x = b->min[0]; x < b->max[0]; ++x)
      {
        uint8_t vox = *(*vd)(x, y, z);
        if (vox < minvox)
        {
          minvox = vox;
        }

        if (vox > maxvox)
        {
          maxvox = vox;
        }

        avg_rgba += { rgbatf[vox * 4], rgbatf[vox * 4 + 1], rgbatf[vox * 4 + 2], rgbatf[vox * 4 + 3] };
        avg += (uint32_t)vox;
      }
    }
  }
  b->minvox = minvox;
  b->maxvox = maxvox;
  const size_t numvox = ((b->max[0] - b->min[0]) * (b->max[1] - b->min[1]) * (b->max[2] - b->min[2]));
  if (numvox > 0)
  {
    b->avg = avg / numvox;
    b->avg_rgba = avg_rgba / virvo::vec4(static_cast<float>(numvox));
  }
}

virvo::vector< 3, size_t > makeBricks(const vvVolDesc* vd, std::vector<Brick*>* bricks, virvo::vector< 3, size_t > const& bricksize,
  const std::vector<float>& rgbatf)
{
  assert(bricks != nullptr && bricks->size() == 0);
  assert(rgbatf.size() == 256 * 4);

  virvo::vector< 3, size_t > numbricks(ceil((float)vd->vox[0] / (float)bricksize[0]),
    (float)ceil(vd->vox[1] / (float)bricksize[1]), (float)ceil(vd->vox[2] / (float)bricksize[2]));

  for (size_t z = 0; z < numbricks[2]; ++z)
  {
    for (size_t y = 0; y < numbricks[1]; ++y)
    {
      for (size_t x = 0; x < numbricks[0]; ++x)
      {
        virvo::vector< 3, size_t > min(x * bricksize[0], y * bricksize[1], z * bricksize[2]);
        virvo::vector< 3, size_t > max(min[0] + bricksize[0], min[1] + bricksize[1], min[2] + bricksize[2]);

        for (size_t d = 0; d < 3; ++d)
        {
          if (max[d] >= vd->vox[d])
          {
            max[d] = vd->vox[d] - 1;
          }
        }
        Brick* b = new Brick(min, max);
        initBrick(vd, b, rgbatf);
        bricks->push_back(b);
      }
    }
  }
  return numbricks;
}

void subdivide(std::shared_ptr< KdTreeNode > node, std::vector< std::shared_ptr< KdTreeNode > >* kdnodes,
    std::vector< std::shared_ptr< KdTreeNode > >* kdleafs, virvo::vector< 3, size_t > const& corner,
    virvo::vector< 3, size_t > const& numbricks, virvo::vector< 3, size_t > const& bricksize, std::vector<Brick*>* bricks)
{
  assert(node != nullptr);

  const static uint8_t threshold = 2;
  size_t bestpos = 0;
  uint8_t besthomogeneity = 0;
  virvo::vec4 bestavgrgba(0.0f);
  size_t bestaxis = 0;
  bool split = false;

  for (size_t axis = 0; axis < 3; ++axis)
  {
    if (numbricks[axis] <= 1)
    {
      continue;
    }

    // inner split positions
    size_t left = corner[axis] * bricksize[axis] + 1 * bricksize[axis];
    size_t right = numbricks[axis] * bricksize[axis] - 1 * bricksize[axis];
    for (size_t splitpos = left; splitpos < right; splitpos += bricksize[axis])
    {
      virvo::vec4 avg_rgba[2] =
      {
        virvo::vec4( 0.0f ),
        virvo::vec4( 0.0f )
      };
      size_t avg[2] = { 0, 0 };
      size_t numleft = 0;
      size_t numright = 0;
      for (auto it = bricks->begin(); it != bricks->end(); ++it)
      {
        if ((*it)->min[axis] > splitpos)
        {
          avg[1] += (*it)->avg;
          avg_rgba[1] += (*it)->avg_rgba;
          ++numright;
        }
        else
        {
          avg[0] += (*it)->avg;
          avg_rgba[0] += (*it)->avg_rgba;
          ++numleft;
        }
      }

      assert(numleft > 0);
      assert(numright > 0);
      avg[0] = static_cast<size_t>( static_cast<float>(avg[0]) / static_cast<float>(numleft) );
      avg[1] = static_cast<size_t>( static_cast<float>(avg[1]) / static_cast<float>(numright) );
      avg_rgba[0] /= virvo::vec4( static_cast<float>(numleft) );
      avg_rgba[1] /= virvo::vec4( static_cast<float>(numright) );

#if 0
      // homogeneity == 0 means identical
      uint8_t homogeneity = static_cast<uint8_t>(abs(avg[1] - avg[0]));//std::cerr << (int)avg[0] << " " << (int)avg[1] << " " << (int)homogeneity << std::endl;
      if (homogeneity > besthomogeneity && homogeneity > threshold)
      {
        besthomogeneity = homogeneity;
        bestpos = splitpos;
        bestaxis = axis;
        split = true;
      }
#endif

      const static virvo::vec4 rgba_threshold(0.4f, 0.4f, 0.4f, 0.4f);
      virvo::vec4 diff_rgba(std::abs(avg_rgba[1][0] - avg_rgba[0][0]), std::abs(avg_rgba[1][1] - avg_rgba[0][1]),
        std::abs(avg_rgba[1][2] - avg_rgba[0][2]), std::abs(avg_rgba[1][3] - avg_rgba[0][3]));
      float sum_diff = diff_rgba[0] + diff_rgba[1] + diff_rgba[2] + diff_rgba[3];
      float bestsum = bestavgrgba[0] + bestavgrgba[1] + bestavgrgba[2] + bestavgrgba[3];
      if (diff_rgba[0] < rgba_threshold[0] && diff_rgba[1] < rgba_threshold[1]
       && diff_rgba[2] < rgba_threshold[2] && diff_rgba[3] < rgba_threshold[3]
       && sum_diff > bestsum)
      {
        bestavgrgba = diff_rgba;
        bestpos = splitpos;
        bestaxis = axis;
        split = true;
      }
    }
  }

  if (split)
  {
    std::vector<Brick*> left;
    std::vector<Brick*> right;
    for (auto it = bricks->begin(); it != bricks->end(); ++it)
    {
      if ((*it)->min[bestaxis] > bestpos)
      {
        right.push_back(*it);
      }
      else
      {
        left.push_back(*it);
      }
    }
    assert(left.size() > 0);
    assert(right.size() > 0);

    virvo::vector< 3, size_t > cornerleft;
    virvo::vector< 3, size_t > numleft;
    virvo::vector< 3, size_t > cornerright;
    virvo::vector< 3, size_t > numright;
    for (size_t axis = 0; axis < 3; ++axis)
    {
      if (axis == bestaxis)
      {
        assert(bestpos % bricksize[axis] == 0);
        cornerleft[axis] = corner[axis];
        numleft[axis] = bestpos / bricksize[axis]; // ?
        cornerright[axis] = cornerleft[axis] + numleft[axis];
        numright[axis] = numbricks[axis] - bestpos / bricksize[axis];
      }
      else
      {
        cornerleft[axis] = corner[axis];
        numleft[axis] = numbricks[axis];
        cornerright[axis] = corner[axis];
        numright[axis] = numbricks[axis];
      }
    }
    node->leftChild = std::make_shared< KdTreeNode >(cornerleft * bricksize, (cornerleft + numleft) * bricksize);
    node->rightChild = std::make_shared< KdTreeNode >(cornerright * bricksize, (cornerright + numright) * bricksize);
    node->leftChild->parent = node;
    node->rightChild->parent = node;
    subdivide(node->leftChild, kdnodes, kdleafs, cornerleft, numleft, bricksize, &left);
    subdivide(node->rightChild, kdnodes, kdleafs, cornerright, numright, bricksize, &right);

    node->firstChildOffset = static_cast<index_t>(kdnodes->size());
    kdnodes->push_back(node->leftChild);
    kdnodes->push_back(node->rightChild);
  }
  else
  {
   kdleafs->push_back(node);
  }
}
}

namespace virvo
{
struct SplatRend::Impl
{
    Impl() : oldquality(1.0f), opacityscale(1.0f) {}

    std::unique_ptr<vvShaderProgram> shader;
    std::unique_ptr<CLProgram> clprogram;

    std::vector<float> rgbatf;

    std::vector<GLuint> texnames;
    GLuint lutname;
    GLuint vertexbuffer;
    GLuint texcoordbuffer;
    GLuint indexbuffer;

    float oldquality;

    float opacityscale;

    size_t pointspernode;
    std::vector< vec4 > points;
    std::vector< vec4 > texcoords;
    std::vector<float> scalars;
    std::vector<float> tf;
    std::vector<GLuint> indices;

    typedef std::shared_ptr< KdTreeNode > node_ptr;

    node_ptr kdroot;
    std::vector< node_ptr > kdnodes;
    std::vector< node_ptr > kdleafs;

    void make_tf(vvVolDesc* vd)
    {
        glGenTextures(1, &lutname);

        size_t lutEntries = 256;
        tf.resize(4 * lutEntries);
        vd->computeTFTexture(lutEntries, 1, 1, &tf[0]);

        if (clprogram)
        {
            updateCLTf(std::move(clprogram), tf, opacityscale);
        }
    }

    void make_vol_tex(vvVolDesc* vd)
    {

        // only one frame supported so far
        texnames.resize(1);
        glGenTextures(1, texnames.data());

        glBindTexture(GL_TEXTURE_3D_EXT, texnames[0]);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
        glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage3D(GL_PROXY_TEXTURE_3D_EXT, 0, GL_LUMINANCE,
            vd->vox[0], vd->vox[1], vd->vox[2], 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
        GLint glwidth;
        glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D_EXT, 0, GL_TEXTURE_WIDTH, &glwidth);
        if (glwidth != 0)
        {
            glTexImage3D(GL_TEXTURE_3D_EXT, 0, GL_LUMINANCE, vd->vox[0], vd->vox[1], vd->vox[2], 0,
                GL_LUMINANCE, GL_UNSIGNED_BYTE, vd->getRaw(0));
        }
        else
        {
            throw std::runtime_error("Could not accommodate volume texture");
        }
    }

    void make_kd_tree(vvVolDesc* vd)
    {
        kdroot.reset();

        std::vector<Brick*> bricks;
        vector< 3, size_t > bricksize(8, 8, 8);
        vector< 3, size_t > numbricks = makeBricks(vd, &bricks, bricksize, rgbatf);
        kdroot = std::make_shared< KdTreeNode >(vector< 3, size_t >(0, 0, 0), numbricks * bricksize);
        kdnodes.clear();
        kdnodes.push_back(kdroot);
        kdleafs.clear();
        subdivide(kdroot, &kdnodes, &kdleafs, vector< 3, size_t >(0, 0, 0), numbricks, bricksize, &bricks);

        VV_LOG(0) << "Num kd-leafs: " << kdleafs.size();

        pointspernode = 50;
    }
};


SplatRend::SplatRend(vvVolDesc* vd, vvRenderState rs)
  : vvRenderer(vd, rs)
  , impl(new virvo::SplatRend::Impl)
{
  if (glewInit() != GLEW_OK)
  {
    VV_LOG(0) << "Could not initialize GLEW";
    return;
  }
    
    impl->clprogram = std::unique_ptr<CLProgram>(initCLProgram());
    impl->make_tf(vd);
    updateTransferFunction();
    impl->make_kd_tree(vd);
    createSamples();
    impl->make_vol_tex(vd);
}

SplatRend::~SplatRend()
{
  glDeleteTextures(1, impl->texnames.data());
  glDeleteTextures(1, &impl->lutname);
  glDeleteBuffers(1, &impl->vertexbuffer);
  glDeleteBuffers(1, &impl->texcoordbuffer);
  glDeleteBuffers(1, &impl->indexbuffer);
}

void SplatRend::renderVolumeGL()
{
  if (!_showBricks)
  {
    if (impl->oldquality != _quality)
    {
      impl->pointspernode = (size_t)(_quality * 50.0f);
      createSamples();
      impl->oldquality = _quality;
    }
    renderSplats();
  }
//  else
  {
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_3D_EXT);
    for (auto it = impl->kdleafs.begin(); it != impl->kdleafs.end(); ++it)
    {
        typedef virvo::basic_aabb< float > boxf;
        boxf aabb(vd->objectCoords((*it)->min), vd->objectCoords((*it)->max));
        auto vertices = compute_vertices(aabb);
      glBegin(GL_LINES);
      glColor3f(1.0f, 1.0f, 1.0f);

          // front
          glVertex3f(vertices[0][0], vertices[0][1], vertices[0][2]);
          glVertex3f(vertices[1][0], vertices[1][1], vertices[1][2]);

          glVertex3f(vertices[1][0], vertices[1][1], vertices[1][2]);
          glVertex3f(vertices[2][0], vertices[2][1], vertices[2][2]);

          glVertex3f(vertices[2][0], vertices[2][1], vertices[2][2]);
          glVertex3f(vertices[3][0], vertices[3][1], vertices[3][2]);

          glVertex3f(vertices[3][0], vertices[3][1], vertices[3][2]);
          glVertex3f(vertices[0][0], vertices[0][1], vertices[0][2]);

          // back
          glVertex3f(vertices[4][0], vertices[4][1], vertices[4][2]);
          glVertex3f(vertices[5][0], vertices[5][1], vertices[5][2]);

          glVertex3f(vertices[5][0], vertices[5][1], vertices[5][2]);
          glVertex3f(vertices[6][0], vertices[6][1], vertices[6][2]);

          glVertex3f(vertices[6][0], vertices[6][1], vertices[6][2]);
          glVertex3f(vertices[7][0], vertices[7][1], vertices[7][2]);

          glVertex3f(vertices[7][0], vertices[7][1], vertices[7][2]);
          glVertex3f(vertices[4][0], vertices[4][1], vertices[4][2]);

          // left
          glVertex3f(vertices[5][0], vertices[5][1], vertices[5][2]);
          glVertex3f(vertices[0][0], vertices[0][1], vertices[0][2]);

          glVertex3f(vertices[3][0], vertices[3][1], vertices[3][2]);
          glVertex3f(vertices[6][0], vertices[6][1], vertices[6][2]);

          // right
          glVertex3f(vertices[1][0], vertices[1][1], vertices[1][2]);
          glVertex3f(vertices[4][0], vertices[4][1], vertices[4][2]);

          glVertex3f(vertices[7][0], vertices[7][1], vertices[7][2]);
          glVertex3f(vertices[2][0], vertices[2][1], vertices[2][2]);
      glEnd();
    }
  }

  if (_boundaries)
  {
    drawBoundingBox(vd->getSize(), vd->pos, _boundColor);
  }

  vvRenderer::renderVolumeGL();
}

void SplatRend::setParameter(ParameterType param, vvParam const& newValue)
{
    switch (param)
    {
    case vvRenderer::VV_SLICEINT:
      if (_interpolation != static_cast< virvo::tex_filter_mode >(newValue.asInt()))
      {
        _interpolation = static_cast< virvo::tex_filter_mode >(newValue.asInt());
        for (size_t f = 0; f < 1 /*vd->frames */; ++f)
        {
          glBindTexture(GL_TEXTURE_3D_EXT, impl->texnames[f]);
          glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
          glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
        }
        updateTransferFunction();
      }
      break;
    default:

        vvRenderer::setParameter(param, newValue);
        break;

    }
}

void SplatRend::updateTransferFunction()
{
  static const size_t lutentries = 256;
  impl->rgbatf.resize(lutentries * 4);
  vd->computeTFTexture(lutentries, 1, 1, impl->rgbatf.data());

  glBindTexture(GL_TEXTURE_2D, impl->lutname);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R_EXT, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, lutentries, 1, 0,
               GL_RGBA, GL_FLOAT, impl->rgbatf.data());
}

void SplatRend::setVolDesc(vvVolDesc* vd)
{
    vvRenderer::setVolDesc(vd);
    impl->make_kd_tree(vd);
    createSamples();
    impl->make_vol_tex(vd);
}

void SplatRend::renderSplats()
{
#if 1
  cl::buffer invModelview = clCreateBuffer(impl->clprogram->context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * 16, NULL, NULL);
  cl::buffer invProjection = clCreateBuffer(impl->clprogram->context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * 16, NULL, NULL);
  cl::buffer viewport = clCreateBuffer(impl->clprogram->context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(virvo::Viewport), nullptr, nullptr);

  auto invmv = inverse(gl::getModelviewMatrix());
  auto invpr = inverse(gl::getProjectionMatrix());
  auto vp = gl::getViewport();

  cl_int err = clEnqueueAcquireGLObjects(impl->clprogram->commands, 1, impl->clprogram->clpbo.ptr(), 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "Error registering pbo";
  }

  err = clEnqueueWriteBuffer(impl->clprogram->commands, invModelview, CL_TRUE, 0, sizeof(float) * 16, invmv.data(), 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(impl->clprogram->commands, invProjection, CL_TRUE, 0, sizeof(float) * 16, invpr.data(), 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(impl->clprogram->commands, viewport, CL_TRUE, 0, sizeof(virvo::Viewport), &vp, 0, nullptr, nullptr);

  cl_mem blockCounter = clCreateBuffer(impl->clprogram->context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, NULL);
  if (!blockCounter)
  {
    VV_LOG(0) << "Error allocating device memory";
  }

  uint bc[1] = { 0 };
  err = clEnqueueWriteBuffer(impl->clprogram->commands, blockCounter, CL_TRUE, 0, sizeof(uint), bc, 0, NULL, NULL);

  cl_uint numspheres = static_cast<cl_uint>(impl->pointspernode);
  size_t blockSize[2] = { 8, 8 };
  size_t width = vvToolshed::getTextureSize(vp[2]);
  size_t height = vvToolshed::getTextureSize(vp[3]);
  size_t gridSize[2] = { virvo::toolshed::iDivUp(width, blockSize[0]), virvo::toolshed::iDivUp(height, blockSize[1]) };
  uint numBlocks = gridSize[0] * gridSize[1];
  uint gridWidth = gridSize[0];

  err = 0;
  int arg = 0;
  err |= clSetKernelArg(impl->clprogram->spherekernel, arg++, sizeof(cl_mem), &impl->clprogram->clpbo);
  err |= clSetKernelArg(impl->clprogram->spherekernel, arg++, sizeof(cl_mem), &impl->clprogram->bvhnodes);
  err |= clSetKernelArg(impl->clprogram->spherekernel, arg++, sizeof(cl_mem), &impl->clprogram->spheres);
  err |= clSetKernelArg(impl->clprogram->spherekernel, arg++, sizeof(numspheres), &numspheres);
  err |= clSetKernelArg(impl->clprogram->spherekernel, arg++, sizeof(cl_mem), &impl->clprogram->scalars);
  err |= clSetKernelArg(impl->clprogram->spherekernel, arg++, sizeof(cl_mem), &impl->clprogram->tf);
  err |= clSetKernelArg(impl->clprogram->spherekernel, arg++, sizeof(cl_mem), &invModelview);
  err |= clSetKernelArg(impl->clprogram->spherekernel, arg++, sizeof(cl_mem), &invProjection);
  err |= clSetKernelArg(impl->clprogram->spherekernel, arg++, sizeof(cl_mem), &viewport);
  err |= clSetKernelArg(impl->clprogram->spherekernel, arg++, sizeof(cl_mem), &blockCounter);
  err |= clSetKernelArg(impl->clprogram->spherekernel, arg++, sizeof(numBlocks), &numBlocks);
  err |= clSetKernelArg(impl->clprogram->spherekernel, arg++, sizeof(gridWidth), &gridWidth);

#if KERNEL_TIMER
  cl_event event;
#endif
  err = clEnqueueNDRangeKernel(impl->clprogram->commands, impl->clprogram->spherekernel, 2, NULL, &gridSize[0], &blockSize[0], 0, NULL,
#if KERNEL_TIMER
  &event
#else
  NULL
#endif
  );

  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "Error executing kernel: " << virvo::cl::errorString(err);
    return;
  }

  err = clEnqueueReleaseGLObjects(impl->clprogram->commands, 1, impl->clprogram->clpbo.ptr(), 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    VV_LOG(0) << "Error unregistering pbo";
  }

  clFinish(impl->clprogram->commands);

  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, impl->clprogram->glpbo);
  glBindTexture(GL_TEXTURE_2D, impl->clprogram->gltex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, vp[2], vp[3], GL_RGBA, GL_FLOAT, NULL);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glEnable(GL_TEXTURE_2D);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glBegin(GL_QUADS);

  glTexCoord2f(0.0, 0.0);
  glVertex3f(-1.0, -1.0, 0.5);

  glTexCoord2f((float)vp[2] / (float)vvToolshed::getTextureSize(vp[2]), 0.0);
  glVertex3f(1.0, -1.0, 0.5);

  glTexCoord2f((float)vp[2] / (float)vvToolshed::getTextureSize(vp[2]), (float)vp[3] / (float)vvToolshed::getTextureSize(vp[3]));
  glVertex3f(1.0, 1.0, 0.5);

  glTexCoord2f(0.0, (float)vp[3] / (float)vvToolshed::getTextureSize(vp[3]));
  glVertex3f(-1.0, 1.0, 0.5);

  glEnd();

  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  glDisable(GL_TEXTURE_2D);
  glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
  glDisable(GL_LIGHTING);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glEnable(GL_TEXTURE_3D_EXT);
  glBindTexture(GL_TEXTURE_3D_EXT, impl->texnames[0]);
  glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

  impl->shader->enable();
  impl->shader->setParameterTex3D("pix3dtex", impl->texnames[0]);
  impl->shader->setParameterTex2D("pixLUT", impl->lutname);

  glBindBuffer(GL_ARRAY_BUFFER, impl->vertexbuffer);
  glVertexPointer(4, GL_FLOAT, 0, NULL);
  glEnableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, impl->texcoordbuffer);
  glTexCoordPointer(4, GL_FLOAT, 0, NULL);
  glEnableClientState(GL_TEXTURE_COORD_ARRAY);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, impl->indexbuffer);

  glPointSize(3.0f);
  glDrawElements(GL_POINTS, impl->indices.size(), GL_UNSIGNED_INT, NULL);

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  impl->shader->disable();
#endif

#if 0
  auto it1 = impl->points.begin();
  auto it2 = impl->texcoords.begin();
  for (; it1 != impl->points.end() && it2 != impl->texcoords.end(); ++it1, ++it2)
  {
    glBegin(GL_POINTS);
      glVertex3f((*it1)[0], (*it1)[1], (*it1)[2]);
      glTexCoord3f((*it2)[0], (*it2)[2], (*it2)[2]);
    glEnd();
  }
#endif
}

void SplatRend::createSamples()
{
  impl->points.clear();
  impl->texcoords.clear();
  impl->scalars.clear();
  genPoints(&impl->points, &impl->texcoords, &impl->scalars, &impl->indices, impl->kdleafs, impl->pointspernode, vd);

  std::vector<cl_float4> spheres;
  for (auto it = impl->points.begin(); it != impl->points.end(); ++it)
  {
    vec3 sample((*it)[0], (*it)[1], (*it)[2]);
    sample[1] = -sample[1];
    sample[2] = -sample[2];
    cl_float4 sphere = { sample[0], sample[1], sample[2], 1.0f };
    spheres.push_back(sphere);
  }

  // copy tree aabb's
  std::vector< box > oldnodes;
  for (auto it = impl->kdnodes.begin(); it != impl->kdnodes.end(); ++it)
  {
    oldnodes.push_back(box((*it)->min, (*it)->max));
  }

  for (auto it = impl->kdleafs.begin(); it != impl->kdleafs.end(); ++it)
  {
    std::shared_ptr< KdTreeNode > node = *it;
    size_t w = node->max.x - node->min.x;
    size_t h = node->max.y - node->min.y;
    size_t d = node->max.z - node->min.z;
    float normradius = std::cbrt((3.0f / 4.0f) * ((w * h * d) / ((float)impl->pointspernode * M_PI)));
    normradius /= impl->opacityscale;

    for (size_t i = node->firstSphereOffset; i < node->firstSphereOffset + impl->pointspernode; ++i)
    {
      spheres[i].s[3] = normradius;
      vec3 center(spheres[i].s[0], spheres[i].s[1], spheres[i].s[2]);
      box aabb(vd->voxelCoords(center - normradius), vd->voxelCoords(center + normradius));
      box tmp = combine(*node, aabb);
      node->min = tmp.min;
      node->max = tmp.max;
    }

    // refit the bvh
    std::shared_ptr< KdTreeNode > parent = node->parent;
    while (parent != nullptr)
    {
        box tmp = combine(*parent, *node);
        parent->min = tmp.min;
        parent->max = tmp.max;
      parent = parent->parent;
    }
  }

  std::vector<float> kdnodes;
  for (auto it = impl->kdnodes.begin(); it != impl->kdnodes.end(); ++it)
  {
    std::shared_ptr< KdTreeNode > node = *it;

    virvo::aabb objaabb(vd->objectCoords(node->min), vd->objectCoords(node->max));

    for (size_t i = 0; i < 3; ++i)
    {
      kdnodes.push_back(objaabb.min[i]);
    }
    if (node->isLeaf())
    {
      kdnodes.push_back((float)node->firstSphereOffset);
    }
    else
    {
      kdnodes.push_back((float)node->firstChildOffset);
    }

    for (size_t i = 0; i < 3; ++i)
    {
      kdnodes.push_back(objaabb.max[i]);
    }
    if (node->isLeaf())
    {
      kdnodes.push_back(1.0f);
    }
    else
    {
      kdnodes.push_back(0.0f);
    }
  }
  updateCLPoints(std::move(impl->clprogram), kdnodes, spheres, impl->scalars);

  // recreate old tree
  auto it1 = impl->kdnodes.begin();
  auto it2 = oldnodes.begin();
  for ( ; it1 != impl->kdnodes.end() && it2 != oldnodes.end(); ++it1, ++it2)
  {
    (*it1)->min = (*it2).min;
    (*it1)->max = (*it2).max;
  }

  std::cerr << impl->scalars.size() << std::endl;
}
} // virvo


/* plugin create */
vvRenderer* createSplatRend(vvVolDesc* vd, vvRenderState const& rs)
{
    return new virvo::SplatRend(vd, rs);
}


