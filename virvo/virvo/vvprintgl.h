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

#ifndef _VVPRINTGL_H_
#define _VVPRINTGL_H_

#include <float.h>

#include "vvexport.h"
#include "vvvecmath.h"
#include "vvx11.h"

//============================================================================
// Class Definition
//============================================================================

/** This class allows 2D billboard type text to be printed on an OpenGL
    canvas.
    @author Jurgen P. Schulze
*/
class VIRVOEXPORT vvPrintGL
{
  private:
    GLuint base;
    GLint glsRasterPos[4];                        ///< stores GL_CURRENT_RASTER_POSITION
    GLfloat glsColor[4];                          ///< stores GL_CURRENT_COLOR
#ifdef HAVE_X11
    static Display* dsp;
#endif
    vvVector4 _fontColor;
    bool _consoleOutput;

    void saveGLState();
    void restoreGLState();

  public:
    vvPrintGL();
    virtual ~vvPrintGL();
    void print(const float, const float, const char *, ...);

    void setFontColor(const vvVector4& fontColor);
};
#endif

//============================================================================
// End of File
//============================================================================