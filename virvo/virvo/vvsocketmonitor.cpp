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

#ifndef _WIN32

#include "vvdebugmsg.h"
#include "vvsocketmonitor.h"

#include <fcntl.h>
#include <iostream>

using std::cerr;
using std::endl;

vvSocketMonitor::vvSocketMonitor()
{
  FD_ZERO(&_readsockfds);
  FD_ZERO(&_writesockfds);
  FD_ZERO(&_errorsockfds);

  _highestSocketNum = -1;
}

vvSocketMonitor::~vvSocketMonitor()
{
  clear();
}

void vvSocketMonitor::addReadFds(const std::vector<vvSocket*>& readfds)
{
  vvDebugMsg::msg(3, "vvSocketMonitor::addReadFds()");

  _readSockets = readfds;

  for (std::vector<vvSocket*>::const_iterator it = _readSockets.begin(); it != _readSockets.end(); ++it)
  {
    vvSocket* socket = (*it);
    FD_SET(socket->get_sockfd(), &_readsockfds);

    if (socket->get_sockfd() > _highestSocketNum)
    {
      _highestSocketNum = socket->get_sockfd();
    }
  }
}

void vvSocketMonitor::addWriteFds(const std::vector<vvSocket*>& writefds)
{
  vvDebugMsg::msg(3, "vvSocketMonitor::addWriteFds()");

  _writeSockets = writefds;

  for (std::vector<vvSocket*>::const_iterator it = _writeSockets.begin(); it != _writeSockets.end(); ++it)
  {
    vvSocket* socket = (*it);
    FD_SET(socket->get_sockfd(), &_writesockfds);

    if (socket->get_sockfd() > _highestSocketNum)
    {
      _highestSocketNum = socket->get_sockfd();
    }
  }
}

void vvSocketMonitor::addErrorFds(const std::vector<vvSocket*>& errorfds)
{
  vvDebugMsg::msg(3, "vvSocketMonitor::addErrorFds()");

  _errorSockets = errorfds;

  for (std::vector<vvSocket*>::const_iterator it = _errorSockets.begin(); it != _errorSockets.end(); ++it)
  {
    vvSocket* socket = (*it);
    FD_SET(socket->get_sockfd(), &_errorsockfds);

    if (socket->get_sockfd() > _highestSocketNum)
    {
      _highestSocketNum = socket->get_sockfd();
    }
  }
}

vvSocket* vvSocketMonitor::wait(const double timeout)
{
  vvDebugMsg::msg(3, "vvSocketMonitor::wait()");

  timeval* tout;
  if(timeout >= 0.0)
  {
    tout = new timeval;
    tout->tv_sec  = static_cast<int>(timeout);
    tout->tv_usec = (timeout - static_cast<int>(timeout)) * 1000000.0;
  }
  else
  {
    tout = NULL;
  }

  int done = select(_highestSocketNum + 1, &_readsockfds, &_writesockfds, &_errorsockfds, tout);
  if (done > 0)
  {
    for (std::vector<vvSocket*>::const_iterator it = _readSockets.begin(); it != _readSockets.end(); ++it)
    {
      vvSocket* socket = (*it);
      if (FD_ISSET(socket->get_sockfd(), &_readsockfds))
      {
        return socket;
      }
    }

    for (std::vector<vvSocket*>::const_iterator it = _writeSockets.begin(); it != _writeSockets.end(); ++it)
    {
      vvSocket* socket = (*it);
      if (FD_ISSET(socket->get_sockfd(), &_writesockfds))
      {
        return socket;
      }
    }

    for (std::vector<vvSocket*>::const_iterator it = _errorSockets.begin(); it != _errorSockets.end(); ++it)
    {
      vvSocket* socket = (*it);
      if (FD_ISSET(socket->get_sockfd(), &_errorsockfds))
      {
        return socket;
      }
    }
  }
  else if(done == 0)
  {
    vvDebugMsg::msg(3, "vvSocketMonitor::wait() timelimit reached.");
    return NULL;
  }
  else if(done == -1)
  {
    vvDebugMsg::msg(2, "vvSocketMonitor::wait() error by select() returned!");
    switch(errno)
    {
    case EAGAIN:
      vvDebugMsg::msg(2, "vvSocketMonitor::wait() The kernel was (perhaps temporarily) unable to allocate the requested number of file descriptors.");
      break;
    case EBADF:
      vvDebugMsg::msg(2, "vvSocketMonitor::wait() One of the descriptor sets specified an invalid descriptor.");
      break;
    case EINTR:
      vvDebugMsg::msg(2, "vvSocketMonitor::wait() A signal was delivered before the time limit expired and before any of the selected events occurred..");
      break;
    case EINVAL:
      vvDebugMsg::msg(2, "vvSocketMonitor::wait() The specified time limit is invalid. One of its components is negative or too large. \n OR: ndfs is greater than FD_SETSIZE and _DARWIN_UNLIMITED_SELECT is not defined.");
      break;
    default:
      vvDebugMsg::msg(2, "vvSocketMonitor::wait() unknown erro occurred.");
      break;
    }
  }

  return NULL;
}

void vvSocketMonitor::clear()
{
  vvDebugMsg::msg(3, "vvSocketMonitor::clear()");
  for (std::vector<vvSocket*>::const_iterator it = _readSockets.begin(); it != _readSockets.end(); ++it)
  {
    delete (*it);
  }
  for (std::vector<vvSocket*>::const_iterator it = _writeSockets.begin(); it != _writeSockets.end(); ++it)
  {
    delete (*it);
  }
  for (std::vector<vvSocket*>::const_iterator it = _errorSockets.begin(); it != _errorSockets.end(); ++it)
  {
    delete (*it);
  }

  FD_ZERO(&_readsockfds);
  FD_ZERO(&_writesockfds);
  FD_ZERO(&_errorsockfds);
}

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
