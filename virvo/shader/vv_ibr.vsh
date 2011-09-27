uniform mat4 reprojectionMatrix;
uniform sampler2D rgbaTex;
uniform sampler2D depthTex;
uniform float width;
uniform float height;

void main(void)
{
  vec2 tc = vec2(gl_Vertex.x/width, gl_Vertex.y/height);
  gl_FrontColor = texture2D(rgbaTex, tc);
  vec4 p = gl_Vertex;
  p.z = texture2D(depthTex, tc).r;
  if(p.z <= 0.)
  {
      p.z = 1.1;
      gl_FrontColor.a = 0.;
  }
  gl_Position = reprojectionMatrix * p;
}
