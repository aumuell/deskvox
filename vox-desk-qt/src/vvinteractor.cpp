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

#include "vvinteractor.h"

using virvo::vec3f;


vvInteractor::vvInteractor()
  : enabled_(true)
  , has_focus_(false)
  , visible_(true)
  , pos_(vec3f(0.0f, 0.0f, 0.0f))
{

}

vvInteractor::~vvInteractor()
{

}

void vvInteractor::setEnabled(bool enabled)
{
  enabled_ = enabled;
}

void vvInteractor::setFocus()
{
  has_focus_ = true;
}

void vvInteractor::clearFocus()
{
  has_focus_ = false;
}

void vvInteractor::setVisible(bool visible)
{
  visible_ = visible;
}

void vvInteractor::setPos(vec3f const& pos)
{
  pos_ = pos;
}

bool vvInteractor::enabled() const
{
  return enabled_;
}

bool vvInteractor::hasFocus() const
{
  return has_focus_;
}

bool vvInteractor::visible() const
{
  return visible_;
}

vec3f vvInteractor::pos() const
{
  return pos_;
}

