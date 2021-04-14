#version 330				// Shader 3.3
precision highp float;		// normal floats, makes no difference on desktop computers

uniform vec3 wLookAt, wRight, wUp;
layout(location = 0) in vec2 cCamWindowVertex;	// Varying input: vp = vertex position is expected in attrib array 0

out vec3 p;

void main() {
    gl_Position = vec4(cCamWindowVertex, 0, 1);		// transform vp from modeling space to normalized device space
    p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
}